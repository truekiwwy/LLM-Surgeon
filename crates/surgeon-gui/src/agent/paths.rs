//! What an agent is told about the open project, API sessions and CLI runs alike: the exact
//! paths it works with (project folder, source model, active working copy) and the rule that
//! it works on its own with them, so that it never stops to ask for a path or a dataset.

use std::path::{Path, PathBuf};

/// The built-in probe sets of the engine (`surgeon_bench`), with what each measures.
const PROBES: &str = "builtin:math, builtin:logic, builtin:general (surgeon_bench eval : \
                      calcul, logique, connaissances, réponses numériques) et builtin:refusals \
                      (surgeon_bench refusal : demandes légitimes souvent refusées à tort)";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPaths {
    pub project: PathBuf,
    /// The source model, read only.
    pub origin: Option<PathBuf>,
    /// The newest working copy (`copies/<id>`), absolute.
    pub copy: Option<PathBuf>,
}

impl ProjectPaths {
    /// Read from the project folder each time: the copy may appear after the session began.
    pub fn of(project: &Path) -> Self {
        let origin = std::fs::read_to_string(project.join("project.surgeon.json"))
            .ok()
            .and_then(|text| crate::studio::project::load_manifest(&text).ok())
            .and_then(|m| m.source)
            .map(|s| s.path);
        let copy = crate::studio::view::newest_copy_in(&project.join("copies"))
            .map(|id| project.join("copies").join(id));
        Self {
            project: project.to_path_buf(),
            origin,
            copy,
        }
    }

    /// The paths and the autonomy rules, for the system prompt and the CLI prompt.
    pub fn brief(&self) -> String {
        let shown = |p: &Option<PathBuf>, none: &str| {
            p.as_ref()
                .map_or(none.to_owned(), |p| p.display().to_string())
        };
        format!(
            "Chemins absolus du projet :\n\
             - project_dir : {project}\n\
             - origin_path : {origin} (modèle d'origine, verrouillé en écriture)\n\
             - working_copy_path : {copy}\n\
             Autonomie : le projet est ouvert et tout ce qu'il faut est ci-dessus. Ne t'arrête \
             pas pour demander un chemin ni un dataset : inspecte directement \
             working_copy_path (surgeon_inspect, surgeon_graph), mesure avec le jeu intégré \
             qui correspond au symptôme, sans fichier : {PROBES}. Enchaîne les étapes sans \
             attendre de validation ; ne pose une question que si un choix irréversible ou \
             réellement ambigu l'exige (les écritures passent de toute façon par une \
             approbation). Chaque outil surgeon_* renvoie un run_id : cite-le dans \
             record_finding.",
            project = self.project.display(),
            origin = shown(&self.origin, "aucun"),
            copy = shown(
                &self.copy,
                "aucune pour l'instant (le studio la crée ; en attendant, lis origin_path)"
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_brief_names_the_exact_paths_and_the_probe_sets() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model");
        std::fs::create_dir_all(&source).unwrap();
        let manifest = serde_json::json!({
            "schema_version": crate::studio::project::MANIFEST_VERSION,
            "name": "p", "created_unix": 0,
            "source": {"path": source, "sha256": "x", "bytes": 0},
            "state": "ready",
        });
        std::fs::write(
            dir.path().join("project.surgeon.json"),
            manifest.to_string(),
        )
        .unwrap();
        let none = ProjectPaths::of(dir.path());
        assert_eq!(none.copy, None);
        std::fs::create_dir_all(dir.path().join("copies").join("copy-1")).unwrap();
        std::fs::create_dir_all(dir.path().join("copies").join("copy-2.partial")).unwrap();
        let paths = ProjectPaths::of(dir.path());
        let copy = dir.path().join("copies").join("copy-1");
        assert_eq!(paths.copy.as_deref(), Some(copy.as_path()));
        let brief = paths.brief();
        for needle in [
            format!("working_copy_path : {}", copy.display()),
            format!("project_dir : {}", dir.path().display()),
            "builtin:refusals".to_owned(),
            "Ne t'arrête pas pour demander un chemin".to_owned(),
        ] {
            assert!(brief.contains(&needle), "{needle} in {brief}");
        }
    }
}
