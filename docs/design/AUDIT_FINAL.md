# Audit final : interface unifiée G5

Branche `feat/gui-g5-unified-ui`, relevé après les neuf passes de design, la passe d'audit
et les redlines des captures. Point de départ : [AUDIT.md](AUDIT.md) ; règles :
[DIRECTION.md](DIRECTION.md).

## 1. Captures

Rendu hors écran par le rasteriseur CPU de l'application, espace de démonstration jetable :

```bash
surgeon-gui --demo-data --scale 1.0 --no-motion --screenshot <png> --screen <écran> --size <LxH>
```

| Écran | 1440×900 | 1920×1080 |
| --- | --- | --- |
| Atelier | `screenshots/after/atelier-1440x900.png` | `screenshots/after/atelier-1920x1080.png` |
| Graphe 3D | `screenshots/after/graph-1440x900.png` | `screenshots/after/graph-1920x1080.png` |
| Analyses | `screenshots/after/analyses-1440x900.png` | `screenshots/after/analyses-1920x1080.png` |
| Agents | `screenshots/after/agents-1440x900.png` | `screenshots/after/agents-1920x1080.png` |
| Diagnostics | `screenshots/after/diagnostics-1440x900.png` | `screenshots/after/diagnostics-1920x1080.png` |

Les captures « avant » restent dans `screenshots/before/`. Chaque image a été ouverte et
relue ; redlines relevées puis corrigées pendant la revue : boîte de dialogue backend
par-dessus la capture, segments inversés dans les en-têtes, accent en petit texte, champs
accolés, panneau Matériel coupé, libellés de l'histogramme sur les barres, formulaire étiré
à 1 186 px, bouton du rapport au ras du bord, barre de titre et table des deltas à 960×600.
**Redlines ouvertes : 0.**

Le contenu de démonstration est réel : un petit modèle écrit, copié, contrôlé et inspecté par
le vrai moteur (journal, événements), le graphe de test qwen2-mini avec sa référence, un agent
scripté qui inspecte, consigne un constat mesuré puis attend une approbation. Le matériel et
le journal technique sont ceux de la machine qui capture.

## 2. Avant / après

| Mesure | Avant | Après |
| --- | --- | --- |
| Destinations de navigation | 9 sur deux niveaux | 5 onglets, un niveau |
| `Frame::group()` | 9 | 0 |
| `Color32::` hors thème et composants | 6 | 0 (`tests/ui_lint.rs`) |
| Couleurs du chrome | 22 RVB + 9 alphas | 17 jetons + 7 paliers de données |
| Tailles de police dans le shell et les vues | 9 | 5 (11, 12, 13, 14, 16) |
| Polices | Ubuntu Light, Hack (défaut egui) | Barlow Regular et Medium, JetBrains Mono, embarquées |
| Spinners | 9 | 0 |
| Animation la plus longue | 450 ms | 120 ms ; 0 avec `--no-motion` |
| Chaînes d'interface localisées | 120 (`Lang::pick`, `desktop/` seul) | 381 clés FR/EN pour le shell, les vues et la 3D, plus les 93 `Lang::pick` de l'assistant |
| Chaînes en dur dans `src/ui/` | non mesuré | 0 phrase (`tests/ui_lint.rs`) |
| Tests du crate | non relevé | 201 (lib 174, `desktop` 8, `packaging` 5, `ui_render` 8, `ui_lint` 3, `ui_i18n` 3) |

## 3. Plafond d'accent

Part des pixels à moins de 24 (Tchebychev, RVB) de `#2773F5` ou `#0946C8`, hors régions de
données (aperçu 3D, vue 3D, histogramme), 1440×900, contrôlée par
`ui_render::no_screen_spends_more_than_three_percent_of_its_pixels_on_the_accent`.

| Écran | Part | Plafond |
| --- | --- | --- |
| Atelier | 0,09 % | 3 % |
| Graphe 3D | 0,13 % | 3 % |
| Analyses | 1,60 % | 3 % |
| Agents | 0,08 % | 3 % |
| Diagnostics | 0,70 % | 3 % |

Analyses porte le seul aplat d'action visible (`Lancer l'analyse`, actif) ; dans l'Atelier et
Agents, l'action primaire est désactivée tant que le formulaire est incomplet.

## 4. Contrastes (WCAG 2.x)

Calculés par `theme::contrast_failures()`, vérifiés par `ui_render` et par `--self-test`
(ligne `interface`). Seuils : texte 7:1, secondaire et états 4,5:1, désactivé et accent 3:1.

| Premier plan | surface | window | canvas | surface_hover | selection |
| --- | --- | --- | --- | --- | --- |
| `text` | 14.64 | 16.20 | 16.33 | 13.54 | 12.32 |
| `text_secondary` | 7.08 | 7.83 | 7.89 | 6.54 | 5.96 |
| `text_disabled` | 3.81 | 4.21 | 4.24 | 3.52 | 3.20 |
| `ok` | 6.81 | 7.54 | 7.59 | 6.30 | 5.73 |
| `warn` | 6.85 | 7.58 | 7.64 | 6.34 | 5.77 |
| `error` | 5.54 | 6.13 | 6.18 | 5.12 | 4.66 |
| `accent` (indicateur) | 4.00 | 4.43 | 4.46 | 3.70 | 3.37 |

`on_action` sur `action` : 7.72 ; sur `action_hover` : 6.24. `data[6]` sur `canvas` : 13.11 ;
`data[5]` : 7.89. Seul changement de jeton pendant l'audit : `error` passe de `#F85149` à
`#FA5C54` (4,34:1 sur une ligne sélectionnée, sous le seuil).

## 5. Logomark

Grille carrée 5×5 à trois intensités de l'accent, tirée de la planche Atelier validée et
tracée en vecteur (`ui/logo.rs`), cellules alignées sur les pixels. Définitive (choix motivé
dans DIRECTION.md, section 5) ; l'icône de fenêtre (`assets/icon.png`) n'a pas été redessinée.

## 6. Polices et licences

| Fichier | Famille | Licence |
| --- | --- | --- |
| `assets/fonts/Barlow-Regular.ttf`, `Barlow-Medium.ttf` | Barlow (Jeremy Tribby) | SIL OFL 1.1, `Barlow-OFL.txt` |
| `assets/fonts/JetBrainsMono-Regular.ttf` | JetBrains Mono | SIL OFL 1.1, `JetBrainsMono-OFL.txt` |

Couverture vérifiée sur la table `cmap` des fichiers (`ui_render::the_embedded_faces_cover_french_and_the_symbols`) :
tout le français dans Barlow, les symboles (`‖ Δ × → ← σ κ · … ≥ ≤ ± °`) dans Barlow ou en
repli dans JetBrains Mono.

## 7. Commits de la refonte

| Commit | Passe |
| --- | --- |
| `feat(gui): unified shell, five views and demo workspace` | assemblage |
| `refactor(gui): spacing pass` | 1 Spacing |
| `refactor(gui): color pass` | 2 Color |
| `refactor(gui): icons pass` | 3 Icons |
| `refactor(gui): copy pass` | 4 Copy |
| `refactor(gui): type pass` | 5 Type |
| `refactor(gui): layout pass` | 6 Layout |
| `refactor(gui): details pass` | 7 Details |
| `feat(gui): visuals pass on the 3D view` | 8 Visuals |
| `refactor(gui): motion pass` | 9 Motion |
| `fix(gui): impeccable audit and hardening pass` | audit et durcissement |
| `fix(gui): final capture redlines and the wizard's spinners` | redlines des captures |

## 8. Limites connues

- Restent hors du système `ui::components` / `i18n` : l'assistant de première installation,
  le canvas de cartes, l'éditeur de patch et le dialogue backend (widgets egui sur le thème
  commun, français ou `Lang::pick`, tailles 10,5 et 11,5). Les descriptions d'outils et les
  messages d'erreur viennent du moteur, en français.
- La vue 3D reste en perspective ; « orthogonal » désigne le repère d'axes et le tracé filaire.
- Le réglage système « réduire les animations » n'est pas lu : seul `--no-motion` coupe les
  transitions.
- L'historique des runs et la fin du journal technique sont relus à chaque frame sur leur
  onglet (commentaires `shortcut:`), sans souci jusqu'à quelques centaines de runs.
- Les tables ne sont pas virtualisées : l'arbre d'un modèle de 70B (plusieurs milliers de
  tenseurs) reste à mesurer.
- Le rendu hors écran est proche de la fenêtre réelle, pas identique au pixel (rasteriseur CPU).
- Hors Windows, le glisser et le redimensionnement de la fenêtre sans bordure dépendent du
  gestionnaire de fenêtres (winit) ; non testés ici.
