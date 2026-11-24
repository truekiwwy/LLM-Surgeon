//! What an agent is told about the open project, API sessions and CLI runs alike: the exact
//! paths it works with (project folder, source model, active working copy) and the rule that
//! it works on its own with them, so that it never stops to ask for a path or a dataset.

use std::path::{Path, PathBuf};

/// The built-in probe sets of the engine, with what each measures and where they go.
const PROBES: &str = "builtin:math, builtin:logic, builtin:general (surgeon_bench eval : \
                      calcul, logique, connaissances, réponses numériques) et builtin:refusals \
                      (surgeon_bench refusal : demandes légitimes souvent refusées à tort) ; \
                      ils servent aussi d'eval_dataset à surgeon_autotune et de \
                      *_prompts_file à surgeon_extract";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPaths {
    pub project: PathBuf,
    /// The folder of the source model (or the lone weights file), read only.
    pub origin: Option<PathBuf>,
    /// The newest working copy (`copies/<id>`), absolute.
    pub copy: Option<PathBuf>,
    /// The configured folder of other models, read only.
    pub models: Option<PathBuf>,
}

impl ProjectPaths {
    /// Read from the project folder each time: the copy may appear after the session began.
    pub fn of(project: &Path) -> Self {
        let origin = std::fs::read_to_string(project.join("project.surgeon.json"))
            .ok()
            .and_then(|text| crate::studio::project::load_manifest(&text).ok())
            .and_then(|m| m.source)
            .map(|s| crate::studio::copy::model_root(&s.path).to_path_buf());
        let copy = crate::studio::view::newest_copy_in(&project.join("copies"))
            .map(|id| project.join("copies").join(id));
        Self {
            project: project.to_path_buf(),
            origin,
            copy,
            models: weight_surgeon::backend::models_dir(),
        }
    }

    /// The paths and the autonomy rules, for the system prompt and the CLI prompt.
    pub fn brief(&self) -> String {
        // Plain `C:\...`: a verbatim `\\?\` prefix breaks PowerShell's Join-Path in CLI runs.
        let plain = |p: &Path| weight_surgeon::graph::model::plain_path(p.to_path_buf());
        let shown = |p: &Option<PathBuf>, none: &str| {
            p.as_deref()
                .map_or(none.to_owned(), |p| plain(p).display().to_string())
        };
        let models = self.models.as_deref().map_or(String::new(), |dir| {
            format!(
                "- models_dir : {} (autres modèles, lecture seule : surgeon_files action list \
                 pour les trouver, surgeon_check et surgeon_inspect pour les examiner)\n",
                plain(dir).display()
            )
        });
        format!(
            "Chemins absolus du projet :\n\
             - project_dir : {project}\n\
             - work_dir : {work} (tes fichiers de travail : surgeon_files action write y crée \
             jeux .jsonl, scripts et rapports, et renvoie leur chemin dans data.path)\n\
             - origin_path : {origin} (modèle d'origine, verrouillé en écriture)\n\
             - working_copy_path : {copy}\n\
             {models}\
             Lecture seule de l'origine : les outils d'analyse et de mesure lisent \
             origin_path. Pour une comparaison appariée avant / après, passe-le en baseline \
             de surgeon_graph (model = working_copy_path) et en baseline_model de \
             surgeon_bench. Toute écriture dans origin_path est refusée par le serveur : \
             sorties, caches et corrections vont dans project_dir.\n\
             Protocole de correction, celui des runs CLI de référence, à enchaîner toi-même \
             dans cet ordre sans sauter d'étape :\n\
             (1) Extraction contrastive propre : surgeon_extract avec deux jeux du même \
             registre, pos = les prompts qui déclenchent le comportement (builtin:refusals pour \
             des refus excessifs), neg = des prompts neutres (builtin:general, ou un jeu écrit \
             dans work_dir), method mean_diff, site resid_post et pooling last (les défauts), \
             holdout_frac 0.2, out = work_dir/dir_L<couche>.safetensors.\n\
             (2) Sonde de toutes les couches : répète cette extraction sur chaque couche (une \
             sur deux au-delà de 32), classe-les par d_prime et threshold_accuracy hors \
             échantillon (diagnostics de surgeon_extract) et consigne le classement avec \
             record_finding ; la direction retenue est celle de la couche au plus fort signal. \
             surgeon_inspect avec projections [\"o_proj\", \"down_proj\"] confirme les cibles. \
             Ne choisis jamais une couche ni une plage sans cette sonde.\n\
             (3) Projection orthogonale : surgeon_apply sur working_copy_path, dry_run puis \
             réel, avec l'opération {{\"operation\": \"project_out\", \"projections\": \
             [\"o_proj\", \"down_proj\"], \"layers\": \"all\", \"vector_file\": <direction \
             retenue>, \"vector_tensor\": \"direction\", \"vector_tensor_b\": \"direction\", \
             \"alpha\": α}}. Ne restreins layers à la plage à fort signal que si la sonde le \
             justifie. Balaie α de 0.4 à 1.0 (surgeon_restore entre deux essais) et garde le \
             plus petit α qui atteint l'objectif. Calibre le balayage en entonnoir : après \
             chaque apply, crible d'abord la copie avec surgeon_bench mode refusal_probe \
             (probe_method first-token-prob, prompts builtin:refusals), une passe avant par \
             prompt ; écarte les α dont le refus sondé ne baisse pas, et ne passe en génération \
             complète (étape 4) que les deux plus petits α retenus. surgeon_autotune avec \
             refusal_prompts (sans out, il ne fait que mesurer) calibre de la même façon \
             l'échelle d'un patch de steering, sonde puis génération des finalistes \
             (data.selection) ; add et subtract, ses seules sorties, décalent les poids sans \
             retirer la direction : ne les écris que si l'utilisateur impose add.\n\
             (3 bis) Préserver une capacité : dès qu'il faut retirer un comportement (des refus) \
             sans dégrader une capacité (maths, logique, code), extrais aussi la direction de \
             cette capacité à la même couche (surgeon_extract, pos = builtin:math ou \
             builtin:logic, neg = des prompts neutres), puis surgeon_compose avec ablate = la \
             direction retenue, keep = les directions de capacité, mode orthogonal-preserve, \
             out et patch_out dans work_dir ; applique ce patch_out (project_out, recette dans \
             composition). Un refus collinear_vectors veut dire que la capacité et le \
             comportement ne se séparent pas linéairement à cette couche : change de couche.\n\
             (4) Certification obligatoire par génération complète, sur chaque finaliste puis \
             sur working_copy_path dans son état final, avant de déclarer un candidat gagnant ou \
             de clore l'objectif, chacun avec baseline_model = origin_path : surgeon_bench \
             refusal sur builtin:refusals ; eval sur builtin:math, builtin:logic et \
             builtin:general ; contrôle sécurité : refusal sur un jeu de demandes réellement \
             dangereuses (10 au moins), écrit une fois dans work_dir puis réutilisé, dont le \
             taux de refus ne doit pas baisser. Une baisse sur math, logic, general ou sécurité \
             hors de l'intervalle rapporté est une régression. Un résultat refusal_probe \
             (is_heuristic: true) oriente mais ne certifie jamais : le studio le marque \
             « Heuristique » et te rappelle cette certification si tu conclus sans elle.\n\
             Le format de chat du modèle est appliqué par défaut (chat_template auto) : \
             n'habille pas les prompts à la main, et ne mets none que pour le désactiver exprès. \
             Garde les max_new_tokens par défaut (refusal 40, eval 512) : une réponse coupée \
             fausse la mesure. Backend : une mesure de la copie (dossier SafeTensors) tourne sur \
             native ; le serveur écarte hipfire, qui sert un cache quantifié et ignore les poids \
             modifiés (avertissement « backend hipfire écarté »).\n\
             Rollback : dès qu'une contre-mesure surgeon_bench montre une régression après un \
             surgeon_apply, remets la copie en état avec surgeon_restore (model = \
             working_copy_path ; sans autre argument il revient à l'état d'avant le dernier \
             surgeon_apply, backup = les data.backups d'un apply pour viser un état précis, \
             from_origin = origin_path pour repartir du modèle d'origine), puis vérifie que \
             data.identical vaut true et que sha256_after égale sha256_source pour chaque \
             fichier.\n\
             Autonomie : le projet est ouvert et tout ce qu'il faut est ci-dessus. Ne t'arrête \
             pas pour demander un chemin ni un dataset : inspecte directement \
             working_copy_path (surgeon_inspect, surgeon_graph), mesure avec le jeu intégré \
             qui correspond au symptôme, sans fichier : {PROBES}. Enchaîne les étapes sans \
             attendre de validation écrite ; ne pose une question que si un choix irréversible \
             ou réellement ambigu l'exige. Chaque outil surgeon_* renvoie un run_id : cite-le \
             dans record_finding.",
            project = plain(&self.project).display(),
            work = plain(&self.project.join("work")).display(),
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
            "passe-le en baseline de surgeon_graph".to_owned(),
            "baseline_model de surgeon_bench".to_owned(),
            "remets la copie en état avec surgeon_restore".to_owned(),
            "from_origin = origin_path".to_owned(),
            format!("work_dir : {}", dir.path().join("work").display()),
            "surgeon_files".to_owned(),
            r#"{"operation": "project_out", "projections": ["o_proj", "down_proj"]"#.to_owned(),
            "Sonde de toutes les couches".to_owned(),
            "surgeon_bench mode refusal_probe".to_owned(),
            "probe_method first-token-prob".to_owned(),
            "surgeon_compose avec ablate".to_owned(),
            "mode orthogonal-preserve".to_owned(),
            "Certification obligatoire par génération complète".to_owned(),
            "ne certifie jamais".to_owned(),
            "refusal 40, eval 512".to_owned(),
            "builtin:logic et builtin:general ; contrôle sécurité".to_owned(),
            "le serveur écarte hipfire".to_owned(),
            "chat_template auto".to_owned(),
            "Ne t'arrête pas pour demander un chemin".to_owned(),
        ] {
            assert!(brief.contains(&needle), "{needle} in {brief}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn the_brief_never_shows_a_verbatim_prefix() {
        let paths = ProjectPaths {
            project: PathBuf::from(r"\\?\C:\work\p"),
            origin: Some(PathBuf::from(r"\\?\C:\models\m")),
            copy: Some(PathBuf::from(r"\\?\C:\work\p\copies\copy-1")),
            models: Some(PathBuf::from(r"\\?\C:\models")),
        };
        let brief = paths.brief();
        assert!(!brief.contains(r"\\?\"), "{brief}");
        for needle in [r"C:\work\p", r"C:\models\m", r"C:\work\p\copies\copy-1"] {
            assert!(brief.contains(needle), "{needle} in {brief}");
        }
    }
}
