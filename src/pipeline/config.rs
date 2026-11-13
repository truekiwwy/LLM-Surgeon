//! `pipeline.json`, version 1: the model, the patch template and its candidates, the
//! pillars with their Dev and Test sets and their gates, the selection rule, the budgets,
//! the seed and the working-copy strategy. Relative paths resolve against the directory of
//! the file. No gate has a default threshold: every margin, gain, drift and confidence is
//! the configuration's, the code has none to justify.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::backend::BackendConfig;
use crate::bench::dataset::DatasetFormat;
use crate::bench::score::Backend;
use crate::extract::ChatTemplate;
use crate::patch::{PatchFile, MAX_ABS_ALPHA};

use super::stats::BinaryGate;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Pipeline de décision d'un patch (pipeline.json, version 1) : candidats évalués sur le Dev, choix sur le Dev, confirmation unique sur le Test. Les chemins relatifs partent du dossier du fichier."
)]
pub struct PipelineConfig {
    #[schemars(range(min = 1, max = 1), description = "Toujours 1.")]
    pub schema_version: u32,
    #[schemars(
        description = "Dossier SafeTensors du modèle d'origine (Qwen2, Llama/Mistral, Qwen3). Jamais modifié : chaque candidat est appliqué à une copie de travail."
    )]
    pub model: PathBuf,
    #[schemars(
        description = "Gabarit de patch (opérations, projections, couches, rang). Ses alpha sont remplacés (candidates.alphas) ou multipliés (candidates.scales) par chaque candidat."
    )]
    pub patch: PatchFile,
    pub candidates: Candidates,
    #[schemars(
        length(min = 1),
        description = "Piliers, évalués dans l'ordre du tableau (l'ordre de coût) ; un candidat s'arrête à sa première porte non GO."
    )]
    pub pillars: Vec<PillarConfig>,
    pub selection: Selection,
    pub budgets: Budgets,
    #[schemars(description = "Graine des intervalles bootstrap (la génération est gloutonne).")]
    pub seed: u64,
    pub working_copy: WorkingCopy,
    #[serde(default)]
    #[schemars(
        description = "Registre append-only des consultations du Test Set ; défaut : test_access.json à côté de ce fichier. Partagé par les jobs : un Test Set déjà consulté est refusé."
    )]
    pub test_registry: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "Moteur de génération des piliers binaires (safety, unblocking, gsm8k) : native (CPU, défaut), mistralrs (mistral.rs sur le dossier SafeTensors), llama-server (GPU via llama.cpp) ou hipfire (démon AMD). Les options de la ligne de commande passent devant, puis la configuration utilisateur (surgeon backend show). Les piliers kl et logits lisent toujours les logits natifs."
    )]
    pub backend: Option<BackendConfig>,
}

/// The candidate values of the patch's `alpha`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Candidates {
    /// Grille : chaque valeur remplace l'alpha de toutes les opérations du gabarit.
    Alphas(Vec<f32>),
    /// Facteurs : chaque valeur multiplie les alpha du gabarit (par exemple un patch produit
    /// par surgeon autotune --out, dont les alpha diffèrent par couche).
    Scales(Vec<f32>),
}

impl Candidates {
    pub fn values(&self) -> &[f32] {
        match self {
            Self::Alphas(v) | Self::Scales(v) => v,
        }
    }

    /// The template with candidate `value` applied to every operation.
    pub fn patch(&self, template: &PatchFile, value: f32) -> PatchFile {
        let mut patch = template.clone();
        for op in &mut patch.operations {
            op.alpha = match self {
                Self::Alphas(_) => value,
                Self::Scales(_) => op.alpha * value,
            };
        }
        patch
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PillarKind {
    /// Sécurité : taux de refus (marqueurs) sur des prompts nuisibles ; succès = refus.
    Safety,
    /// Déblocage : taux de réponse sur des prompts bénins sur-refusés ; succès = pas de refus.
    Unblocking,
    /// Capacité : exactitude Pass@1 sur un jeu GSM8K / MATH ; succès = réponse juste.
    Gsm8k,
    /// KL(avant‖après) moyenne par prompt sur des prompts neutres (dérive continue).
    Kl,
    /// Part des positions dont le token le plus probable change, par prompt (dérive continue).
    Logits,
}

impl PillarKind {
    pub fn is_binary(self) -> bool {
        matches!(self, Self::Safety | Self::Unblocking | Self::Gsm8k)
    }

    /// Relative cost: one forward pass per prompt for the drifts, generation for the rest.
    pub fn cost_rank(self) -> u8 {
        match self {
            Self::Kl | Self::Logits => 0,
            Self::Gsm8k => 1,
            Self::Unblocking | Self::Safety => 2,
        }
    }

    /// The command of `bench` this pillar reproduces, and its metric.
    pub fn metric(self) -> &'static str {
        match self {
            Self::Safety => "taux de refus (bench refusal, marqueurs), apparié",
            Self::Unblocking => "taux de non-refus (bench refusal, marqueurs), apparié",
            Self::Gsm8k => "exactitude Pass@1 (bench eval --backend native), appariée",
            Self::Kl => "KL(avant‖après) moyenne par prompt (bench dump + bench logits)",
            Self::Logits => "part de top-1 changés par prompt (bench dump + bench logits)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PillarConfig {
    #[schemars(length(min = 1), description = "Identifiant unique du pilier.")]
    pub id: String,
    pub kind: PillarKind,
    #[schemars(
        description = "Jeu Dev : sert au choix du candidat. Prompts (safety, unblocking, kl, logits : tableau JSON, JSONL ou texte) ou dataset (gsm8k)."
    )]
    pub dataset_dev: PathBuf,
    #[schemars(
        description = "Jeu Test : consulté une seule fois, sur le candidat retenu ; contenu différent du Dev."
    )]
    pub dataset_test: PathBuf,
    pub gate: Gate,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "Génération : tokens max par réponse (défauts de bench : 256 refusal, 512 eval)."
    )]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    #[schemars(description = "safety, unblocking : habillage des prompts (défaut chatml).")]
    pub chat_template: Option<ChatTemplate>,
    #[serde(default)]
    #[schemars(
        description = "gsm8k : fichier de gabarit avec {question} (défaut : celui de bench eval)."
    )]
    pub prompt_template: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "gsm8k : format des réponses de référence (défaut auto).")]
    pub format: Option<DatasetFormat>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "gsm8k : premières questions retenues (défaut : toutes)."
    )]
    pub limit: Option<u32>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "kl, logits : tokens max par prompt du dump (défaut de bench dump)."
    )]
    pub max_tokens: Option<u32>,
}

/// The decision rule of a pillar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Gate {
    /// Non-infériorité : borne basse de l'IC de (après − avant) > −margin.
    NonInferiority {
        #[schemars(range(min = 0.0, max = 1.0))]
        margin: f64,
        #[schemars(description = "Niveau bilatéral de l'intervalle, dans ]0.5, 1[.")]
        confidence: f64,
    },
    /// Supériorité : borne basse de l'IC de (après − avant) > min_gain.
    Superiority {
        #[schemars(range(min = 0.0, max = 1.0))]
        min_gain: f64,
        #[schemars(description = "Niveau bilatéral de l'intervalle, dans ]0.5, 1[.")]
        confidence: f64,
    },
    /// Dérive maximale (kl, logits) : borne haute de l'IC bootstrap de la dérive moyenne ≤ value.
    MaxDrift {
        #[schemars(range(min = 0.0))]
        value: f64,
        #[schemars(description = "Niveau bilatéral de l'intervalle, dans ]0.5, 1[.")]
        confidence: f64,
    },
}

impl Gate {
    pub fn confidence(self) -> f64 {
        match self {
            Self::NonInferiority { confidence, .. }
            | Self::Superiority { confidence, .. }
            | Self::MaxDrift { confidence, .. } => confidence,
        }
    }

    /// The oriented threshold of a binary gate; `None` for a drift gate.
    pub fn binary(self) -> Option<BinaryGate> {
        match self {
            Self::NonInferiority { margin, confidence } => Some(BinaryGate {
                threshold: -margin,
                confidence,
            }),
            Self::Superiority {
                min_gain,
                confidence,
            } => Some(BinaryGate {
                threshold: min_gain,
                confidence,
            }),
            Self::MaxDrift { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    #[schemars(
        description = "Pilier qui départage les candidats GO sur le Dev : meilleur effet observé ; à égalité, le plus petit |valeur|, puis l'ordre de la liste."
    )]
    pub pillar: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Budgets {
    #[schemars(
        range(min = 1),
        description = "Temps maximal du job, en secondes (reprises cumulées) ; au-delà : arrêt, verdict INCONCLUSIVE."
    )]
    pub max_wall_time_secs: u64,
    #[schemars(
        range(min = 1),
        description = "Nombre maximal de candidats ; une liste plus longue est refusée."
    )]
    pub max_candidates: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum WorkingCopy {
    /// Copie complète du modèle pour chaque candidat (disque : 1 modèle ; E/S : 1 copie par candidat).
    Copy,
    /// Une copie, puis pour chaque candidat : apply avec sauvegarde des fichiers touchés,
    /// mesure, restauration (disque : 1 modèle + fichiers touchés ; E/S : 2 fois les
    /// fichiers touchés par candidat).
    RestoreFromBackup,
}

/// What is wrong with a configuration, as one message per problem.
pub fn validate(config: &PipelineConfig, allow_disable_safety: bool) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    if config.schema_version != SCHEMA_VERSION {
        issues.push(format!(
            "schema_version {} : seule la version {SCHEMA_VERSION} existe",
            config.schema_version
        ));
    }
    let values = config.candidates.values();
    if values.is_empty() {
        issues.push("candidates : au moins une valeur".into());
    }
    if values.len() > config.budgets.max_candidates as usize {
        issues.push(format!(
            "candidates : {} valeurs pour budgets.max_candidates = {}",
            values.len(),
            config.budgets.max_candidates
        ));
    }
    if values
        .iter()
        .any(|v| !v.is_finite() || v.abs() > MAX_ABS_ALPHA)
    {
        issues.push(format!(
            "candidates : valeurs finies et de valeur absolue ≤ {MAX_ABS_ALPHA}"
        ));
    }
    let mut seen = HashSet::new();
    for v in values {
        if !seen.insert(v.to_bits()) {
            issues.push(format!("candidates : valeur {v} en double"));
        }
    }
    if config.budgets.max_wall_time_secs == 0 || config.budgets.max_candidates == 0 {
        issues.push("budgets : valeurs >= 1".into());
    }
    if let Some(b) = config.backend.as_ref().and_then(|b| b.backend) {
        if !matches!(
            b,
            Backend::Native | Backend::MistralRs | Backend::LlamaServer | Backend::Hipfire
        ) {
            issues.push(format!(
                "backend : {} non géré par le pipeline (native, mistralrs, llama-server ou hipfire)",
                b.label()
            ));
        }
    }
    let mut ids = HashSet::new();
    for p in &config.pillars {
        if p.id.trim().is_empty()
            || !p
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            issues.push(format!(
                "pilier « {} » : identifiant de lettres, chiffres, _ ou - seulement",
                p.id
            ));
        }
        if !ids.insert(p.id.as_str()) {
            issues.push(format!("pilier « {} » en double", p.id));
        }
        let c = p.gate.confidence();
        if !(c > 0.5 && c < 1.0) {
            issues.push(format!("pilier {} : confidence dans ]0.5, 1[", p.id));
        }
        match (p.kind.is_binary(), p.gate) {
            (true, Gate::MaxDrift { .. }) => issues.push(format!(
                "pilier {} : une porte max_drift ne s'applique qu'à kl ou logits",
                p.id
            )),
            (false, Gate::NonInferiority { .. } | Gate::Superiority { .. }) => {
                issues.push(format!(
                    "pilier {} : kl et logits prennent une porte max_drift",
                    p.id
                ))
            }
            _ => {}
        }
        let bounded = |x: f64, max: f64| x.is_finite() && (0.0..=max).contains(&x);
        let ok = match p.gate {
            Gate::NonInferiority { margin, .. } => bounded(margin, 1.0),
            Gate::Superiority { min_gain, .. } => bounded(min_gain, 1.0),
            Gate::MaxDrift { value, .. } => bounded(value, f64::MAX),
        };
        if !ok {
            issues.push(format!(
                "pilier {} : seuil de porte fini et positif (margin, min_gain ≤ 1)",
                p.id
            ));
        }
        let misplaced = |field: &str, set: bool, kinds: &str| {
            set.then(|| format!("pilier {} : {field} ne s'applique qu'à {kinds}", p.id))
        };
        let refusal = matches!(p.kind, PillarKind::Safety | PillarKind::Unblocking);
        let drift = !p.kind.is_binary();
        issues.extend(
            [
                misplaced(
                    "chat_template",
                    p.chat_template.is_some() && !refusal,
                    "safety, unblocking",
                ),
                misplaced(
                    "prompt_template",
                    p.prompt_template.is_some() && p.kind != PillarKind::Gsm8k,
                    "gsm8k",
                ),
                misplaced(
                    "format",
                    p.format.is_some() && p.kind != PillarKind::Gsm8k,
                    "gsm8k",
                ),
                misplaced(
                    "limit",
                    p.limit.is_some() && p.kind != PillarKind::Gsm8k,
                    "gsm8k",
                ),
                misplaced(
                    "max_new_tokens",
                    p.max_new_tokens.is_some() && drift,
                    "safety, unblocking, gsm8k",
                ),
                misplaced("max_tokens", p.max_tokens.is_some() && !drift, "kl, logits"),
            ]
            .into_iter()
            .flatten(),
        );
        if p.dataset_dev == p.dataset_test {
            issues.push(format!(
                "pilier {} : dataset_dev et dataset_test identiques",
                p.id
            ));
        }
    }
    if !config.pillars.iter().any(|p| p.kind == PillarKind::Safety) && !allow_disable_safety {
        issues.push(
            "aucun pilier safety : la porte de sécurité est obligatoire (--allow-disable-safety \
             en ligne de commande seulement)"
                .into(),
        );
    }
    match config
        .pillars
        .iter()
        .find(|p| p.id == config.selection.pillar)
    {
        None => issues.push(format!(
            "selection.pillar « {} » n'est pas un pilier",
            config.selection.pillar
        )),
        Some(p) if !p.kind.is_binary() => issues.push(format!(
            "selection.pillar « {} » : un pilier safety, unblocking ou gsm8k (effet apparié)",
            p.id
        )),
        Some(_) => {}
    }
    match issues.is_empty() {
        true => Ok(()),
        false => Err(issues),
    }
}

/// Every path of the configuration made absolute against `base` (the file's directory).
pub fn resolve(config: &mut PipelineConfig, base: &Path) {
    let abs = |p: &mut PathBuf| {
        if p.is_relative() {
            *p = base.join(&*p);
        }
    };
    abs(&mut config.model);
    for op in &mut config.patch.operations {
        abs(&mut op.vector_file);
    }
    for p in &mut config.pillars {
        abs(&mut p.dataset_dev);
        abs(&mut p.dataset_test);
        if let Some(t) = p.prompt_template.as_mut() {
            abs(t);
        }
    }
    let registry = config
        .test_registry
        .get_or_insert_with(|| PathBuf::from("test_access.json"));
    abs(registry);
    if let Some(b) = config.backend.as_mut() {
        b.llama_cpp_dir.iter_mut().for_each(abs);
        b.mistralrs_dir.iter_mut().for_each(abs);
        b.mistralrs_bin.iter_mut().for_each(abs);
        b.hipfire_bin.iter_mut().for_each(abs);
    }
}

/// Every file the run reads, with the name of its field (for the MCP sandbox).
pub fn inputs(config: &PipelineConfig) -> Vec<(String, PathBuf)> {
    let mut out = vec![("model".to_string(), config.model.clone())];
    for (i, op) in config.patch.operations.iter().enumerate() {
        out.push((
            format!("patch.operations[{i}].vector_file"),
            op.vector_file.clone(),
        ));
    }
    for p in &config.pillars {
        out.push((
            format!("pillars.{}.dataset_dev", p.id),
            p.dataset_dev.clone(),
        ));
        out.push((
            format!("pillars.{}.dataset_test", p.id),
            p.dataset_test.clone(),
        ));
        if let Some(t) = &p.prompt_template {
            out.push((format!("pillars.{}.prompt_template", p.id), t.clone()));
        }
    }
    if let Some(r) = &config.test_registry {
        out.push(("test_registry".into(), r.clone()));
    }
    // They hold programs that get run.
    if let Some(b) = &config.backend {
        for (field, path) in [
            ("llama_cpp_dir", &b.llama_cpp_dir),
            ("mistralrs_dir", &b.mistralrs_dir),
            ("mistralrs_bin", &b.mistralrs_bin),
            ("hipfire_bin", &b.hipfire_bin),
        ] {
            if let Some(p) = path {
                out.push((format!("backend.{field}"), p.clone()));
            }
        }
    }
    out
}

/// JSON Schema (draft-07) of `pipeline.json`, for `surgeon schema --kind pipeline`.
pub fn schema() -> Value {
    let generator = SchemaSettings::draft07().into_generator();
    serde_json::to_value(generator.into_root_schema_for::<PipelineConfig>()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PipelineConfig {
        serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "model": "m",
            "patch": {"schema_version": 1, "operations": [{"id": "o", "operation": "subtract",
                "alpha": 1.0, "layers": "all", "projections": ["down_proj"], "vector_file": "v"}]},
            "candidates": {"alphas": [0.5, 1.0]},
            "pillars": [
                {"id": "kl", "kind": "kl", "dataset_dev": "a", "dataset_test": "b",
                 "gate": {"max_drift": {"value": 0.1, "confidence": 0.9}}},
                {"id": "safety", "kind": "safety", "dataset_dev": "c", "dataset_test": "d",
                 "gate": {"non_inferiority": {"margin": 0.05, "confidence": 0.95}}}
            ],
            "selection": {"pillar": "safety"},
            "budgets": {"max_wall_time_secs": 60, "max_candidates": 2},
            "seed": 1,
            "working_copy": "copy"
        }))
        .unwrap()
    }

    fn issues(c: &PipelineConfig, allow: bool) -> String {
        validate(c, allow).err().unwrap_or_default().join(" | ")
    }

    #[test]
    fn a_valid_configuration_and_its_paths() {
        let mut c = config();
        validate(&c, false).unwrap();
        resolve(&mut c, Path::new("/base"));
        assert_eq!(c.model, Path::new("/base").join("m"));
        assert_eq!(
            c.test_registry.as_deref(),
            Some(Path::new("/base").join("test_access.json").as_path())
        );
        assert_eq!(inputs(&c).len(), 1 + 1 + 4 + 1);
        let scaled = Candidates::Scales(vec![0.5]).patch(&c.patch, 0.5);
        assert_eq!(scaled.operations[0].alpha, 0.5);
        assert_eq!(
            Candidates::Alphas(vec![0.2])
                .patch(&c.patch, 0.2)
                .operations[0]
                .alpha,
            0.2
        );
    }

    #[test]
    fn the_safety_pillar_is_mandatory_unless_the_cli_disables_it() {
        let mut c = config();
        c.pillars.retain(|p| p.kind != PillarKind::Safety);
        c.selection.pillar = "kl".into();
        assert!(issues(&c, false).contains("aucun pilier safety"));
        assert!(!issues(&c, true).contains("aucun pilier safety"));
    }

    #[test]
    fn inconsistent_configurations_are_refused() {
        let mut c = config();
        c.pillars[0].gate = Gate::NonInferiority {
            margin: 0.1,
            confidence: 0.9,
        };
        c.pillars[1].dataset_test = c.pillars[1].dataset_dev.clone();
        c.pillars[1].max_tokens = Some(8);
        c.selection.pillar = "kl".into();
        c.candidates = Candidates::Alphas(vec![0.5, 0.5, 1.0]);
        let text = issues(&c, false);
        for needle in [
            "kl et logits prennent une porte max_drift",
            "dataset_dev et dataset_test identiques",
            "max_tokens ne s'applique qu'à kl, logits",
            "selection.pillar « kl »",
            "en double",
            "budgets.max_candidates = 2",
        ] {
            assert!(text.contains(needle), "{needle}: {text}");
        }
        let mut c = config();
        c.pillars[1].gate = Gate::NonInferiority {
            margin: 0.1,
            confidence: 1.0,
        };
        assert!(issues(&c, false).contains("confidence"));
    }

    #[test]
    fn the_schema_is_draft_07_and_strict() {
        let schema = schema();
        assert_eq!(schema["$schema"], "http://json-schema.org/draft-07/schema#");
        assert_eq!(schema["additionalProperties"], false);
        let unknown = serde_json::from_value::<PipelineConfig>(serde_json::json!({
            "schema_version": 1, "frobnicate": true
        }));
        assert!(unknown.is_err());
    }
}
