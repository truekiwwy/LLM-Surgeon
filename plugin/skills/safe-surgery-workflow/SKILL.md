---
name: safe-surgery-workflow
description: Démarche obligatoire pour toute chirurgie de poids avec les outils MCP surgeon (surgeon_check, surgeon_extract, surgeon_apply, surgeon_bench...). À suivre dès qu'on modifie, oriente (steering), fusionne ou évalue un modèle local, même pour un essai rapide. Impose check d'abord, travail sur une copie, dry_run avant apply, sauvegarde gardée et mesure avant toute conclusion.
---

# Chirurgie sûre avec surgeon

Le serveur MCP `surgeon` modifie des poids **en place**. Une erreur coûte un modèle de
plusieurs gigaoctets. Suis ces étapes dans l'ordre, sans en sauter une, même si
l'utilisateur est pressé. Si une étape échoue, arrête-toi et rapporte l'erreur
(`error.kind`, `exit_code`) au lieu d'improviser.

## 1. Vérifier avant tout

- Appelle `surgeon_check` sur le modèle. Si le verdict est `insufficient`, arrête-toi et
  donne les chiffres (RAM nécessaire et disponible).
- Appelle `surgeon_inspect` pour connaître les couches et les formes des matrices MLP
  avant de choisir `layers`, `projections` et `axis`.

## 2. Travailler sur une copie

- Ne patche jamais le modèle original. Le chemin passé à `surgeon_apply`, `surgeon_lora`
  (mode `merge`) ou `surgeon_bench` doit être une **copie** placée dans le dossier
  autorisé.
- S'il n'existe pas de copie, demande à l'utilisateur d'en faire une, ou crée-la toi-même
  seulement s'il t'y autorise explicitement.
- Avant toute modification, prends la référence de mesure : `surgeon_bench` en mode `dump`
  sur la copie encore intacte, avec un fichier de prompts que tu réutiliseras à l'étape 6.

## 3. Préparer le patch

- Pour un vecteur de direction, utilise `surgeon_extract` (`method` : `mean_diff` par
  défaut, `svd` si les prompts sont appariés, `leace` pour `project_out`). Lis `d_prime` et
  `threshold_accuracy` : une séparation faible (avertissement `d′ < 1`) donne une direction
  peu fiable, dis-le.
- Commence avec un `alpha` petit et un petit nombre de couches.

## 4. dry_run avant apply

- Appelle **toujours** `surgeon_apply` avec `dry_run: true` d'abord. Vérifie dans le
  rapport la liste des tenseurs visés et `vector_len`.
- Seulement ensuite, appelle `surgeon_apply` sans `dry_run`, sur la copie.
- Même règle pour `surgeon_lora` en mode `merge` : `dry_run: true` d'abord.

## 5. Ne jamais désactiver la sauvegarde

- N'écris jamais `backup: false` dans un patch et ne propose jamais de relancer le serveur
  avec `--allow-no-backup`.
- Si `surgeon_apply` échoue avec le code 3, le modèle est intact (rollback). Avec le code 4,
  c'est CRITIQUE : préviens l'utilisateur immédiatement et indique les chemins de
  sauvegarde du rapport.

## 6. Mesurer après le patch

- Mesure avec `surgeon_bench` : mode `dump` sur la copie patchée avec le même fichier de
  prompts que la référence, puis mode `logits` entre les deux dumps (KL, changements
  top-1). Ajoute `ppl` ou `eval` quand llama.cpp est disponible ; sans llama.cpp, `eval`
  avec `backend: "native"` et `refusal` (taux de refus, avec `baseline_model`) génèrent en
  Rust pur sur un dossier Qwen2.
- Pour une non-régression de génération, utilise `surgeon_eval` sur la copie patchée.

## 7. Ne rien annoncer sans mesure

- Chaque affirmation sur l'effet du patch cite le chiffre mesuré et l'outil qui l'a
  produit (par exemple « KL moyen 0,012, top-1 inchangé à 97 % (surgeon_bench logits) »).
- Si une mesure n'a pas été faite, écris « non mesuré ». Ne déduis jamais l'effet d'un
  patch de son seul `alpha` ou de la réussite de `surgeon_apply`.
