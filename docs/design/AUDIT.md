# Audit initial du slop : `crates/surgeon-gui`

Relevé fait AVANT toute modification de l'interface, sur la branche `feat/gui-g5-unified-ui`
(base `aab3791`). Pile réelle : `egui` / `eframe` 0.36.2, backend `glow` (OpenGL), rendu 3D
dessiné en formes 2D par le `Painter` (`viewport.rs`, pas de pipeline GL propre), rendu hors
écran par un rasteriseur CPU (`raster.rs`).

Captures « avant » : `docs/design/screenshots/before/` (`home`, `wizard`, `graph`, `agents`,
`diagnostics`, 1440×900, échelle 1.0). L'écran `Projets` (ancêtre d'`Analyses`) n'avait pas
d'option de capture : il est décrit depuis le code (`studio/view.rs`).

Constat de départ : la branche ne compilait pas (`studio/view.rs:450`, champ `read_only`
manquant après la fusion G3/G4). Corrigé en une ligne avant les captures.

## 1. État des briques

| Brique | État | Conséquence pour l'interface |
| --- | --- | --- |
| G2 projets / jobs / événements (`studio/`) | fusionnée | Analyses et l'historique lisent le vrai gestionnaire de jobs et le ledger |
| G3 agents (`agent/`) | fusionnée | Agents, cartes d'approbation et journal des preuves sont réels |
| G4 3D en direct (`studio/live.rs`, `viewport.rs`) | fusionnée | Graphe 3D, rejeu et couche de run sont réels |

Aucun état « Non disponible dans ce build » n'est donc requis pour G2, G3, G4.

## 2. Inventaire chiffré

| Mesure | Valeur | Détail |
| --- | --- | --- |
| `Frame::group()` | 9 | `agent/view.rs` 3, `desktop/wizard.rs` 5, `desktop/mod.rs` 1 |
| Autres constructeurs de cadre | 1 | `canvas/view.rs` |
| `Color32::` hors thème | 6 | `editor/mod.rs` 3, `canvas/view.rs` 1, `desktop/mod.rs` 1, `studio/live.rs` 1 |
| `Color32::` dans `theme.rs` | 20 | 22 couleurs RVB distinctes, plus 9 variantes par `gamma_multiply` |
| Valeurs d'espacement et de taille distinctes | 52 | histogramme ci-dessous |
| Sites d'espacement en dur | 81 | `app.rs` 18, `wizard.rs` 16, `desktop/mod.rs` 14, `agent/view.rs` 10, `canvas/view.rs` 8, `backend_dialog.rs` 6, `editor/view.rs` 4, `studio/view.rs` 4, `viewport.rs` 1 |
| Tailles de police distinctes | 9 | 10.5, 11, 11.5, 12, 12.5, 14, 16, 26, 30 |
| `CollapsingHeader` par couche | 2 sites (`app.rs`) | un en-tête et N widgets par couche, à chaque frame |
| Listes virtualisées (`show_rows`) | 0 | 11 `ScrollArea` dessinent tout leur contenu |
| Spinners | 9 sites | `app.rs` 1, `desktop/mod.rs` 2, `wizard.rs` 3, `studio/view.rs` 3 |
| Animations | 1 | fondu de disposition 3D de 450 ms (`app.rs:955`) |
| Séparateurs `ui.separator()` | 29 | utilisés comme structure à la place d'une grille |
| Chaînes d'interface en dur | ~300 | `app.rs` 55, `agent/view.rs` 54, `wizard.rs` 74, `canvas/view.rs` 43, `studio/view.rs` 38, `desktop/mod.rs` 31 |
| Chaînes bilingues (`Lang::pick`) | 120 | seulement `desktop/` ; le reste est en français seul |
| Emojis | 0 | mais des glyphes de police servent d'icônes : `×` (fermer), `→` `←` (outils), `•` (puces), `|<` `<` `>` `>|` (transport) |
| Ombres | 0 | `window_shadow` et `popup_shadow` déjà à `NONE` |
| `Visuals::dark()` résiduel | 0 appel direct | mais `theme::apply` ne fixe que 20 champs : le reste vient du thème sombre par défaut d'egui |
| Polices | par défaut d'egui | Ubuntu Light, Hack, deux polices d'emojis (`default_fonts`) |

Histogramme des valeurs d'espacement et de taille (valeur × occurrences) :

```
0×5  0.3  0.7  0.75  0.8  1  2×3  3×3  4×5  6×17  7  8×12  10×5  12×4  14×5  18
28×2  30×2  32×2  40  60×3  64  70×2  108  110×3  120×2  150  160×2  172  180  190
200×2  220  240  255  295  300×3  320  330  420  430  460  480  520  600  620  680
800  900  1440
```

La valeur la plus fréquente est 6 (hors de toute grille de 4 ou de 8). Les largeurs de
panneaux (255, 295, 330) ne sont multiples ni de 4 ni de 8.

## 3. Constats

Chaque constat est suivi dans les redlines de la passe indiquée.

| Id | Écran / fichier | Constat | Passe |
| --- | --- | --- | --- |
| S-001 | Graphe 3D (`app.rs` `top_bar`) | La barre du haut porte 18 éléments sur une ligne ; ils se chevauchent à 1440 px (capture `graph.png` : « arêtes » sous « détection », « Colonne » sous « Constellation ») | 6 |
| S-002 | toutes | Deux niveaux de navigation dans la même barre (`Accueil / Atelier / Diagnostics` puis `Studio / Graphe 3D / Split / Projets / Agents`) : 9 destinations, aucune hiérarchie | 6 |
| S-003 | toutes | Espacements : 52 valeurs, dominante à 6 ; hauteurs de barres 28, 30, 32 ; panneaux 255, 295, 330 | 1 |
| S-004 | Graphe 3D | Les trois colonnes ne partagent pas de grille : 255 / fluide / 295, bornes 240-270 et 280-310 | 1 |
| S-005 | toutes | Palette : cyan de sélection `#3FB8D4`, magenta agent `#D75FD7`, 5 couleurs de rôles ANSI (bleu, pervenche, ambre, sable, rouge) : 7 teintes saturées dans le chrome et la 3D | 2 |
| S-006 | 3D, inspecteur | 9 niveaux d'alpha par `gamma_multiply` : dégradés continus dans les données (arêtes, Δ) | 2, 8 |
| S-007 | `theme.rs` | `Visuals` partiels : hover, focus, sélection de texte, barres de défilement, curseur de saisie hérités d'egui | 2 |
| S-008 | hors thème | 6 `Color32::` en dur (`editor/mod.rs`, `canvas/view.rs`, `desktop/mod.rs`, `studio/live.rs`) | 2 |
| S-009 | Diagnostics, Accueil | Badges à crochets `[   OK    ]` `[manquant ]` en monospace coloré, largeur fixe par remplissage d'espaces | 3 |
| S-010 | rejeu, chronologie | Transport en glyphes de police `|<` `<` `>` `>|`, fermeture `×` | 3 |
| S-011 | Agents, canvas | Puces `•` et flèches `→` `←` tirées de la police comme icônes | 3 |
| S-012 | toutes | Aucune icône de fenêtre propre : barre de titre native du système au-dessus de la barre de navigation | 3, 7 |
| S-013 | Agents | Bloc de 5 lignes orange en texte courant (« Claude Code et Codex tournent… ») : un avertissement permanent utilisé comme documentation | 4 |
| S-014 | Projets | Libellés techniques bruts (`{out} = dossier de sortie du run`, `p0`, `unix 1731…`) et verdicts au format `Debug` (`Ok`, `Failed`) | 4 |
| S-015 | toutes | Français seul hors `desktop/` : la bascule FR/EN ne change que l'accueil, l'assistant et Diagnostics | 4 |
| S-016 | toutes | Mêmes termes en double : « Diagnostics » (navigation et titre), « Agent » (onglet, titre), « Annuler » avec trois sens | 4 |
| S-017 | toutes | 9 tailles de police, dont 10.5 et 11 sur du texte courant secondaire ; Ubuntu Light en corps | 5 |
| S-018 | inspecteur | Nombres en monospace mais clés en proportionnelle 12.5 sur lignes zébrées : deux rythmes verticaux | 5 |
| S-019 | Graphe 3D (`layer_panel`) | Un `CollapsingHeader` par couche, tous les nœuds dessinés ; index des couches (`BTreeMap`) et filtre recalculés à chaque frame | 6 |
| S-020 | Projets | `ui.columns(3)` sans redimensionnement ; formulaire, file et historique dans des `ScrollArea` indépendantes sans titre | 6 |
| S-021 | Accueil, Diagnostics | Colonne centrée de 680 px sur 1440 : une application réduite à une page web étroite | 6 |
| S-022 | Agents, assistant, Diagnostics | 9 `Frame::group` : cartes encadrées, dont une carte d'approbation à bordure orange pleine | 7, C |
| S-023 | toutes | Coins 2 et 3 mélangés (`window_corner_radius` 3), bordures 1 pt non alignées sur le pixel à 125 % et 150 % | 7 |
| S-024 | toutes | Aucun anneau de focus maîtrisé ; navigation clavier limitée à haut/bas dans la liste des couches | 7 |
| S-025 | Graphe 3D | Projection en perspective, nœuds pleins colorés par rôle, arêtes à alpha continu : un nuage de points, pas un tracé filaire | 8 |
| S-026 | Graphe 3D | Couleur seule pour le rôle des nœuds (les états de run ont déjà un glyphe) | 8 |
| S-027 | Projets, assistant | 9 spinners ; fondu de disposition 3D de 450 ms ; `ProgressBar::animate(true)` | 9 |
| S-028 | toutes | Aucune prise en compte du réglage « réduire les animations » du système | 9 |
| S-029 | Atelier | Aucun indicateur de la copie de travail active ni de la protection de l'original hors de l'onglet Projets | C |
| S-030 | toutes | États vides muets (« aucun projet ouvert », « aucun graphe ») : ils ne disent pas quoi faire ensuite | C |

## 4. Ce qui est déjà sain

- Les couleurs passent presque toutes par `theme.rs` (6 écarts) ; aucune ombre, aucun dégradé
  de chrome, aucun emoji.
- Les états de run en 3D portent déjà un glyphe et un mot, jamais la couleur seule.
- Le moteur reste pur : le GUI est un crate à part avec son propre `Cargo.lock`.
- Les données affichées sont réelles (aucun contenu factice dans le chemin de production).

Le slop est donc léger sur la couleur et lourd sur la structure (navigation, grille,
densité, micro-copie, i18n) : la refonte porte d'abord sur le shell et les composants.

## 5. Suivi des constats (après refonte)

État au commit de livraison de `feat/gui-g5-unified-ui`. Le détail chiffré est dans
[AUDIT_FINAL.md](AUDIT_FINAL.md).

| Id | Statut | Ce qui a changé |
| --- | --- | --- |
| S-001 | corrigé | Barre de titre de 40 : marque, cinq onglets, mode, fenêtre ; les contrôles 3D vivent dans l'en-tête du panneau 3D |
| S-002 | corrigé | Un seul niveau : Atelier, Graphe 3D, Analyses, Agents, Diagnostics ; cartes et éditeur dans l'Atelier (D-04) |
| S-003 | corrigé dans `ui/` | Grille de 8 pour les zones, paddings {4, 8, 12, 16, 24} ; testé par `layout::tests` |
| S-004 | corrigé | Colonnes 280 / fluide / 320, bornes 240-400 et 280-480, pas de 8, persistées |
| S-005 | corrigé | Deux bleus à rôle fixe, une échelle de données à sept paliers ; plus de couleur de rôle |
| S-006 | partiel | Les arêtes gardent un alpha continu (donnée, région exclue du plafond d'accent) |
| S-007 | corrigé | `theme::style()` fixe tous les états depuis les jetons ; `ui_render` vérifie qu'ils en viennent |
| S-008 | corrigé | 0 `Color32::` hors `ui/theme.rs` et `ui/components.rs` (`tests/ui_lint.rs`) |
| S-009 | corrigé | Point dessiné plus mot (`components::status`) |
| S-010 | corrigé | Transport en icônes vectorielles du jeu `ui/icons.rs` |
| S-011 | corrigé | Lignes d'outils du chat derrière des icônes dessinées, colonne de 16 |
| S-012 | corrigé | Fenêtre sans bordure système, boutons dessinés, glisser, double-clic, 4 bords et 4 coins |
| S-013 | corrigé | Plus de bloc d'avertissement permanent : la politique est un panneau à droite |
| S-014 | partiel | Verdicts en mots localisés ; les descriptions des outils viennent du schéma du moteur (français) |
| S-015 | partiel | Shell, vues et 3D bilingues (381 clés) ; restent en `Lang::pick` ou français seul : assistant, canvas de cartes, éditeur de patch, dialogue backend, messages du moteur |
| S-016 | corrigé | Un terme par sens ; « Annuler » ne désigne plus que l'annulation |
| S-017 | corrigé dans `ui/` | Cinq tailles (11, 12, 13, 14, 16) ; Barlow et JetBrains Mono embarquées ; les modules hérités gardent 10,5 et 11,5 |
| S-018 | corrigé | `components::kv` : clé secondaire, valeur mono à droite, un seul pas de 24 |
| S-019 | corrigé | Arbre dessiné par lignes, couches repliées par défaut ; reste un recalcul par frame (rapide à 30 tenseurs, à mesurer sur un 70B) |
| S-020 | corrigé | Analyses : formulaire typé à mesure de 720, file et historique dans un tableau à colonnes |
| S-021 | corrigé | L'écran d'accueil disparaît (D-02) |
| S-022 | corrigé | 0 `Frame::group` ; panneaux à une bordure, jamais imbriqués |
| S-023 | corrigé | Rayon 2 partout, bordures de 1 pixel physique alignées sur la grille de pixels |
| S-024 | corrigé | Anneau de focus d'accent ; `Ctrl+1..5`, `Ctrl+K`, flèches et Échap testés |
| S-025 | corrigé | Tracé filaire : nœuds en contour, repère d'axes orthogonal ; la projection reste en perspective |
| S-026 | corrigé | Rôle lu à la forme (cercle, carré, losange) |
| S-027 | corrigé | 0 spinner ; morphose 3D de 120 ms ; barre de progression sans boucle |
| S-028 | partiel | `--no-motion` coupe toute transition ; le réglage système n'est pas lu |
| S-029 | corrigé | Copie active et nombre d'écritures dans l'explorateur et la barre d'état (D-06) |
| S-030 | corrigé | Chaque état vide dit ce qui manque puis quoi faire |
