//! The command line and what a path handed to the app means (arguments, a second launch,
//! a drop on the window).

use std::path::{Path, PathBuf};

use clap::Parser;

use crate::canvas::EXTENSION;

#[derive(Debug, Parser)]
#[command(
    name = "surgeon-gui",
    version,
    about = "LLM Surgery Studio : accueil, assistant d'installation, canvas de cartes, éditeur de patchs et graphe des poids en 3D"
)]
pub struct Cli {
    /// graph.json, graph.ndjson (suivi en direct), `-` (NDJSON sur l'entrée standard), un
    /// workspace *.surgeon-studio.json ou un dossier. Sans argument : l'écran d'accueil.
    #[arg(value_name = "PATH")]
    pub inputs: Vec<PathBuf>,
    /// Dossier des projets du Studio (défaut : celui des réglages, sinon ~/Documents/weight-surgeon).
    #[arg(long, value_name = "DOSSIER")]
    pub workspace: Option<PathBuf>,
    /// Vérifie le moteur, l'initialisation graphique, l'écriture du workspace et le studio
    /// (projets, copies, jobs, rejeu, journal), puis quitte (code 0 si tout va bien, 1 sinon).
    #[arg(long)]
    pub self_test: bool,
    /// Rend l'écran courant hors écran dans ce PNG, puis quitte.
    #[arg(long, value_name = "PNG")]
    pub screenshot: Option<PathBuf>,
    /// Avec --screenshot : home (défaut), wizard, diagnostics, graph (la vue 3D du graphe).
    #[arg(long, value_name = "SCREEN", default_value = "home")]
    pub screen: String,
    /// Rejoue le events.ndjson d'un run sur la vue 3D du graphe ouvert (barre de lecture).
    #[arg(long, value_name = "EVENTS")]
    pub replay: Option<PathBuf>,
    /// Efface les réglages du bureau (settings.json) : l'assistant se relance au prochain
    /// démarrage. Les composants installés et le config.toml du moteur sont conservés.
    #[arg(long)]
    pub reset_settings: bool,
}

/// What one path means to the app.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    /// `graph.json`, `graph.ndjson` or `-`: handled by [`crate::source::open`].
    Graph(PathBuf),
    /// A `*.surgeon-studio.json` canvas workspace.
    Studio(PathBuf),
    /// A folder: becomes the workspace folder.
    Folder(PathBuf),
    Unknown(PathBuf),
}

pub fn classify(path: &Path) -> Item {
    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase());
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase());
    match (path.as_os_str() == "-", &name, &ext) {
        (true, _, _) => Item::Graph(path.to_path_buf()),
        _ if path.is_dir() => Item::Folder(path.to_path_buf()),
        (_, Some(n), _) if n.ends_with(EXTENSION) => Item::Studio(path.to_path_buf()),
        (_, _, Some(e)) if e == "json" || e == "ndjson" => Item::Graph(path.to_path_buf()),
        _ => Item::Unknown(path.to_path_buf()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_classified_by_what_they_are() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(classify(Path::new("-")), Item::Graph("-".into()));
        assert!(matches!(
            classify(Path::new("a/graph.json")),
            Item::Graph(_)
        ));
        assert!(matches!(classify(Path::new("a/B.NDJSON")), Item::Graph(_)));
        assert!(matches!(
            classify(Path::new("x.surgeon-studio.json")),
            Item::Studio(_)
        ));
        assert!(matches!(classify(dir.path()), Item::Folder(_)));
        assert!(matches!(classify(Path::new("notes.txt")), Item::Unknown(_)));
    }

    #[test]
    fn the_legacy_arguments_still_parse() {
        for args in [["surgeon-gui", "graph.json"], ["surgeon-gui", "-"]] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(cli.inputs.len(), 1);
        }
        assert!(Cli::try_parse_from(["surgeon-gui"])
            .unwrap()
            .inputs
            .is_empty());
        let cli =
            Cli::try_parse_from(["surgeon-gui", "--screenshot", "a.png", "--self-test"]).unwrap();
        assert!(cli.self_test && cli.screenshot.is_some());
        let cli = Cli::try_parse_from(["surgeon-gui", "--workspace", "projets"]).unwrap();
        assert_eq!(cli.workspace, Some("projets".into()));
    }
}
