# Direction : Weight Surgeon, interface native

Source de vérité de la composition : `docs/design/mockups/` (deux planches validées).
Ce document fixe ce que les neuf passes et l'affinage doivent respecter.

## 1. Les cinq lignes

- **Produit.** Un atelier de chirurgie des poids de LLM locaux : inspecter un modèle,
  cibler un tenseur, appliquer un patch sur une copie, mesurer, sans jamais toucher
  l'original.
- **Caractère.** Dense, calme, exact. Un instrument d'ingénierie bas niveau (CAO, IDE
  natif), jamais décoratif.
- **Monde.** La table à dessin technique et le banc de mesure : traits fins, cotes,
  lettrage normalisé, fond sombre de poste de travail.
- **Moments.** Deux seulement : le graphe 3D filaire orthogonal (le modèle vu comme une
  pièce de CAO) et l'histogramme filaire de ‖ΔW‖. Toute la richesse visuelle est là ; le
  chrome autour est neutre.
- **Signature.** Le logomark carré en grille de blocs et les tracés filaires orthogonaux.
  Rien d'autre ne porte de forme propre.

## 2. Règle de l'accent

Deux bleus, deux rôles, jamais interchangeables.

- `#2773F5` (accent primaire) marque **un seul focus d'attention par région** :
  soulignement de l'onglet actif (barre de titre), élément sélectionné (arbre, 3D),
  segment de mode actif, curseur de saisie et anneau de focus clavier.
- `#0946C8` (accent d'action) remplit **une seule action primaire par écran**, texte blanc
  (7,7:1). Les autres boutons sont secondaires (surface et bordure) ou fantômes.
- L'accent n'est jamais du petit texte courant : `#2773F5` fait 4,0:1 sur `#161B22` et
  4,4:1 sur `#0E0F14`, sous le seuil de 4,5:1. Il sert aux remplissages, soulignements
  et indicateurs (seuil de 3:1 respecté).
- États : OK, avertissement et erreur sont toujours un point dessiné plus un mot
  (`● OK`, `● Manquant`), jamais la couleur seule.

**Plafond de pixels d'accent.** Mesure : part des pixels dont la couleur est à moins de 24
(distance de Tchebychev sur RVB) de `#2773F5` ou `#0946C8`, hors régions de données (canevas
3D, histogramme) dont l'échelle est du contenu. Plafond fixé à **3 %** par écran. Raison :
le bouton primaire le plus large (Analyses, 1440×900) fait environ 0,9 % de l'écran ; avec
le logomark, le soulignement d'onglet, le segment de mode et une ligne sélectionnée, un écran
sain reste sous 2 % ; 3 % laisse une marge pour une carte d'approbation ouverte et échoue dès
qu'un second aplat d'accent apparaît. Les valeurs mesurées sont dans `AUDIT_FINAL.md`, et le
test `ui_captures` les contrôle.

## 3. Volontairement absent

Dégradés, halos, ombres (fenêtre et popups compris), verre, icônes décoratives, emojis,
badges encadrés, cartes imbriquées, spinners, animations de plus de 120 ms, texte en
dégradé, couleur de rôle par nœud. Aucune illustration : les seuls visuels sont des
données.

## 4. Jetons

| Jeton | Valeur | Usage |
| --- | --- | --- |
| `window` | `#0E0F14` | fond de fenêtre, entre les panneaux |
| `canvas` | `#0D0E12` | canevas 3D, histogramme, champs de saisie |
| `surface` | `#161B22` | panneaux |
| `surface_hover` | `#1C222B` | survol de ligne et de bouton secondaire |
| `selection` | `#1B2942` | ligne sélectionnée |
| `border` | `#2A3444` | bordures structurelles, 1 px physique |
| `separator` | `#1B2028` | séparateurs internes |
| `accent` | `#2773F5` | focus d'attention |
| `action` | `#0946C8` | action primaire (survol `#0B52E6`) |
| `text` | `#E6EDF3` | texte principal (≥ 7:1) |
| `text_secondary` | `#9AA7B8` | libellés, méta (≥ 4,5:1) |
| `text_disabled` | `#6B7788` | désactivé, indices (≥ 3:1) |
| `ok` / `warn` / `error` | `#3FB950` / `#D29922` / `#FA5C54` | états (≥ 4,5:1), toujours avec un mot |
| `data[0..7]` | sept paliers d'un seul bleu | échelles de données, par quantiles |

Les ratios exacts sont calculés par le test `theme_contrast` (WCAG 2.x) et reportés dans
`AUDIT_FINAL.md`.

**Espacements.** La consigne d'origine « 8 / 12 / 16 / 24 » mélange des multiples de 8 et
de 4. Choix retenu et testé :

- dimensions de layout (largeur des colonnes, hauteur des barres et des lignes, marges
  entre zones) : multiples de **8** ;
- paddings internes : échelle **{4, 8, 12, 16, 24}** ; 4 et 12 n'existent qu'à l'intérieur
  d'un composant (texte dans un bouton, écart icône-libellé), jamais comme dimension de
  zone.

Rayon ≤ 2. Bordure = 1 pixel physique, rectangles alignés sur la grille de pixels pour
`pixels_per_point` ∈ {1,0 ; 1,25 ; 1,5 ; 2,0}.

**Typographie.** Cinq tailles : 11 (méta), 12 (données mono), 13 (corps), 14 (titres de
panneau, onglets), 16 (marque, titres de dialogue).

- Sans-serif : **Barlow** (Regular et Medium, SIL OFL 1.1). Lignée DIN : un lettrage
  construit géométriquement pour la signalisation et le dessin technique, étroit, lisible
  à 13 px, qui couvre le français. C'est le caractère « table à dessin » du monde choisi ;
  une grotesque d'interface générique ne le porterait pas.
- Monospace : **JetBrains Mono** Regular (SIL OFL 1.1), chiffres tabulaires : noms de
  tenseurs, formes, normes, empreintes, journaux.
- Titres de panneau en capitales (`TENSEURS`, `INSPECTEUR`) : repris des maquettes
  validées, sans interlettrage ajouté. C'est le lettrage d'un cartouche de plan, limité
  aux titres de panneau ; libellés, boutons et onglets restent en casse de phrase.

## 5. Composition

Une fenêtre sans bordure système. Barre de titre de 40 : logomark, `WEIGHT SURGEON`,
`Atelier v{version}`, cinq onglets, boutons de fenêtre (le sélecteur `[SCALPING]` /
`[ADVANCE]` est dans l'en-tête de l'Atelier depuis G6). Corps en trois zones séparées par 8 :
colonne gauche (selon l'onglet, repliable par `Ctrl+B`), centre (travail), colonne droite
(inspection et suivi). Barre d'état de 24. Le premier lancement couvre les trois zones d'une
page d'assistant en quatre étapes.

| Écran | Gauche | Centre | Droite |
| --- | --- | --- | --- |
| Atelier | contexte, arbre de tenseurs | ciblage : histogramme et seuil, table des cibles (ou cartes, patch JSON en `[ADVANCE]`) | opération : formulaire, puis run et journal |
| Graphe 3D | idem | vue 3D, rejeu, cadrage sur les cibles | inspecteur, chronologie |
| Analyses | idem | analyse, formulaire typé, budget, lancement | historique de tous les runs, filtré |
| Agents | projet, conversations | session : `connexion · modèle`, fil, approbations | contrat, preuves |
| Système | sections, relancer l'assistant | dépendances par rôle, fournisseurs IA, mises à jour, journal | matériel, connexion choisie, rapport |

Composition G6 et ses décisions (D-11 à D-29) : [G6_PLAN.md](G6_PLAN.md).

**Logomark.** Les deux planches montrent deux marques différentes : une grille carrée 5×5
à trois tons (planche Atelier) et un damier 5×3 (planche à quatre écrans). Le brief demande
un logomark carré : la grille 5×5 de la planche Atelier est retenue et tracée en vecteur
(`ui/logo.rs`). Elle n'est donc pas provisoire.

## 6. Approches considérées

1. **Restyler l'existant** (garder `Accueil / Atelier / Diagnostics` et les cinq modes,
   changer le thème). Écarté : la navigation à deux niveaux et la barre à 18 éléments
   (S-001, S-002) sont le premier défaut, un thème ne les corrige pas.
2. **Nouveau shell, vues réécrites sur l'état existant** (retenu). `ui/` porte les jetons,
   les composants, le shell et cinq vues ; l'état et la logique (`studio`, `agent`,
   `source`, `viewport`, `diag`) ne bougent pas. Les vues ne font que dessiner et câbler.
3. **Moteur de rendu 3D GL dédié** (shaders, MSAA). Écarté : le `Painter` d'egui tessellise
   déjà chaque trait en quads antialiasés en espace écran, sans `glLineWidth` ; un second
   pipeline n'apporterait qu'un risque de compatibilité GL.

## 7. Journal des décisions

| # | Décision | Raison |
| --- | --- | --- |
| D-01 | Aucune question posée ; les étapes d'entretien de `superpowers` et `impeccable` sont remplacées par ce document | consigne du brief (autonomie complète) |
| D-02 | L'écran d'accueil disparaît ; ouverture de fichier par dépôt, palette (`Ctrl+K`) et colonne gauche ; langue dans la barre d'état ; mises à jour dans Diagnostics | cinq onglets seulement, plus de page centrée de 680 px (S-021) |
| D-03 | L'onglet `Projets` est réparti : projet et copie dans la colonne gauche, lancement dans Analyses et Atelier, historique dans la colonne droite | une seule place pour le contexte (S-029) |
| D-04 | Le canvas de cartes et l'éditeur de patch JSON restent accessibles par la palette, dans le centre de l'Atelier | fonctions existantes conservées sans sixième onglet |
| D-05 | Le rôle d'un nœud 3D se lit à sa forme (cercle, carré, losange), plus à sa couleur | couleur réservée aux données et à la sélection (S-005, S-026) |
| D-06 | Copie active = dossier le plus récent de `copies/` ; « modifiée » = au moins une écriture réussie du ledger sur cette copie | calculé depuis le ledger, jamais déclaré |
| D-07 | ~~`[ADVANCE]` lance l'agent en mode `Demander`~~ remplacée par D-14 (G6) : le mode ne règle que la profondeur de l'Atelier, l'agent vit dans Agents | politique G3 inchangée, aucune auto-approbation |
| D-08 | ~~l'assistant garde `Lang::pick`~~ remplacée par D-29 (G6) : l'assistant est sous `src/ui/`, couvert par le lint, sans `Lang::pick` | périmètre vérifiable, sans exception |
| D-09 | Police sans-serif : Barlow plutôt qu'une grotesque d'interface | caractère DIN du monde choisi |
| D-10 | Pas de MSAA demandé au contexte GL | l'antialiasing vient du feathering d'epaint ; MSAA réduit la compatibilité sans gain mesurable |
