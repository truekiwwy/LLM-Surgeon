LLM Surgery Studio (weight-surgeon) - Windows
=============================================

FR
--
surgeon-gui.exe   L'application (accueil, assistant de premier lancement, studio, graphe 3D).
surgeon.exe       Le même moteur en ligne de commande (surgeon --help).

Premier lancement : double-clic sur surgeon-gui.exe. L'assistant vérifie votre machine et
propose d'installer llama.cpp (GPU Vulkan ou CPU). Rien n'est téléchargé sans votre accord.

Windows SmartScreen : tant que l'exécutable n'est pas signé, Windows peut afficher
« Windows a protégé votre PC ». Cliquez sur « Informations complémentaires » puis
« Exécuter quand même », après avoir comparé le SHA-256 du fichier avec SHA256SUMS (la page de
la release). Voir docs/DESKTOP.md.

Mode portable : si le dossier contient un fichier portable.txt, tout (réglages, composants,
journaux) reste dans le sous-dossier data\ à côté de l'exécutable. Rien n'est écrit dans
votre profil. Pour utiliser surgeon.exe avec la même configuration :
  set WEIGHT_SURGEON_CONFIG=<ce dossier>\data\config.toml

Désinstallation : « Applications installées » de Windows (version installateur), ou supprimez
le dossier (version portable). Les composants téléchargés sont dans
%LOCALAPPDATA%\weight-surgeon (hors mode portable).

EN
--
surgeon-gui.exe   The app (home, first-run wizard, studio, 3D graph).
surgeon.exe       The same engine on the command line (surgeon --help).

First run: double-click surgeon-gui.exe. The wizard checks your machine and offers to install
llama.cpp (Vulkan GPU or CPU). Nothing is downloaded without your consent.

Windows SmartScreen: while the executable is unsigned, Windows may show "Windows protected
your PC". Click "More info" then "Run anyway", after comparing the file's SHA-256 with
SHA256SUMS (on the release page). See docs/DESKTOP.md.

Portable mode: if the folder holds a portable.txt file, everything (settings, components,
logs) stays in the data\ subfolder next to the executable. Nothing is written to your
profile. To use surgeon.exe with the same configuration:
  set WEIGHT_SURGEON_CONFIG=<this folder>\data\config.toml

Uninstall: Windows "Installed apps" (installer version), or delete the folder (portable).
Downloaded components live in %LOCALAPPDATA%\weight-surgeon (outside portable mode).
