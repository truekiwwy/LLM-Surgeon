//! What a session is for, chosen when it starts and kept with its conversation:
//! `Advance` (the agent scans the whole model and finds the problems itself) or `Scalping`
//! (the user names a precise symptom and the agent goes straight to the likely tensors).
//! The brief goes to the agent with the system prompt; the approval policy is unchanged and
//! applied by the studio, so a brief never asks the agent to stop and wait for a validation.

use serde::{Deserialize, Serialize};
use weight_surgeon::patch::LayerSelection;

use crate::ui::i18n::tf;

/// A symptom the user can name in `Scalping`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Symptom {
    Refusals,
    Repetition,
    Hallucination,
    Language,
    Reasoning,
    Other,
}

impl Symptom {
    pub const ALL: [Symptom; 6] = [
        Symptom::Refusals,
        Symptom::Repetition,
        Symptom::Hallucination,
        Symptom::Language,
        Symptom::Reasoning,
        Symptom::Other,
    ];

    /// The built-in probe set that measures the symptom (`surgeon_bench`, no file).
    pub fn probe(self) -> &'static str {
        match self {
            Symptom::Refusals => "builtin:refusals (surgeon_bench mode refusal)",
            Symptom::Reasoning => "builtin:math puis builtin:logic (surgeon_bench mode eval)",
            Symptom::Hallucination => "builtin:general (surgeon_bench mode eval)",
            Symptom::Repetition => {
                "builtin:general (surgeon_bench mode eval, champ loop_detected de chaque item)"
            }
            Symptom::Language | Symptom::Other => {
                "builtin:general (surgeon_bench mode eval), ou un jeu adapté écrit dans work_dir \
                 avec surgeon_files"
            }
        }
    }

    /// Where the agent looks first, for the brief.
    fn lead(self) -> &'static str {
        match self {
            Symptom::Refusals => {
                "refus excessifs : direction de refus du flux résiduel, extraite par contraste \
                 (builtin:refusals contre des prompts neutres), sondée sur toutes les couches, \
                 retirée par project_out sur o_proj et down_proj ; mesure avec surgeon_bench \
                 (refusal)"
            }
            Symptom::Repetition => {
                "répétitions et boucles : têtes d'attention des dernières couches, normes de \
                 lm_head et des embeddings"
            }
            Symptom::Hallucination => {
                "hallucinations : couches MLP du milieu et de la fin (gate, up, down_proj), \
                 conditionnement et rang effectif"
            }
            Symptom::Language => {
                "langue ou format de sortie : embeddings, lm_head et premières couches"
            }
            Symptom::Reasoning => {
                "raisonnement et calcul : couches MLP du milieu, écart à une référence si le \
                 projet en a une ; mesure avec surgeon_bench (eval)"
            }
            Symptom::Other => {
                "autre : déduis les composants probables de la description et construis le jeu \
                 de mesure qui la reproduit"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "symptom", rename_all = "snake_case")]
pub enum Mission {
    Advance,
    Scalping(Symptom),
}

impl Mission {
    /// Appended to the system prompt.
    pub fn brief(self) -> String {
        match self {
            Mission::Advance => "Mission : scan complet (Advance). L'utilisateur ne sait pas \
                 où sont les problèmes du modèle : trouve-les toi-même.\n\
                 1. Vérifie le modèle (surgeon_check, surgeon_inspect) puis construis le \
                 graphe chirurgical complet avec surgeon_graph, sur une copie de copies/.\n\
                 2. Cherche les anomalies : normes et rangs effectifs hors de la distribution \
                 de leur rôle, conditionnement extrême, couches ou experts atypiques, écarts \
                 à une référence s'il y en a une, et des mesures de comportement en lecture \
                 avec les jeux intégrés (surgeon_bench eval sur builtin:math, builtin:logic, \
                 builtin:general ; refusal sur builtin:refusals).\n\
                 3. Rends un rapport classé du plus grave au moins grave : pour chaque \
                 problème, les tenseurs en cause, la mesure (record_finding avec run_id et \
                 valeur exacte) et la correction que tu proposes. Montre le plus grave avec \
                 show_in_3d.\n\
                 4. Ensuite seulement, traite les corrections une par une avec le protocole \
                 de correction (extraction, sonde des couches, project_out, validation \
                 multi-benchmarks) ; restaure la copie si une mesure régresse. L'approbation \
                 éventuelle passe par le studio : ne l'attends pas en texte."
                .into(),
            Mission::Scalping(symptom) => format!(
                "Mission : scalping. L'utilisateur a signalé un problème précis ; sa \
                 description est dans son message. Symptôme : {}. Jeu de mesure : {}.\n\
                 Ne fais pas de scan complet du graphe : mesure le symptôme (surgeon_bench en \
                 lecture), confirme ou infirme l'hypothèse avec record_finding, montre la cible \
                 avec show_in_3d, puis déroule le protocole de correction sans sauter d'étape : \
                 la sonde des couches désigne la cible, jamais une supposition. Restaure la \
                 copie si une mesure régresse. L'approbation éventuelle passe par le studio : \
                 ne l'attends pas en texte.",
                symptom.lead(),
                symptom.probe()
            ),
        }
    }
}

/// The patch operation the advanced (CLI) panel of the form can force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// The protocol's choice (project_out).
    #[default]
    Auto,
    ProjectOut,
    Add,
}

/// The residual-stream writers the panel can force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tensors {
    #[default]
    Both,
    OProj,
    DownProj,
}

impl Tensors {
    /// The `projections` of a patch operation.
    pub fn projections(self) -> &'static str {
        match self {
            Tensors::Both => r#"["o_proj", "down_proj"]"#,
            Tensors::OProj => r#"["o_proj"]"#,
            Tensors::DownProj => r#"["down_proj"]"#,
        }
    }
}

/// What the user forces from the advanced (CLI) panel, kept with the conversation like the
/// mission: the parts of the patch the agent must not choose. An empty `layers` or `alpha`
/// stays the protocol's (the probe, the alpha sweep).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Forced {
    pub operation: Operation,
    pub tensors: Tensors,
    /// `all`, `12-15` or `3,5`: the `layers` of a patch operation.
    pub layers: String,
    pub alpha: String,
}

impl Forced {
    /// Why these values cannot go to the agent, if they cannot.
    pub fn check(&self) -> Result<(), String> {
        let layers = self.layers.trim();
        if !layers.is_empty() {
            layers
                .parse::<LayerSelection>()
                .map_err(|e| tf("forced.layers.invalid", &[("error", &e)]))?;
        }
        let alpha = self.alpha.trim();
        match alpha.parse::<f32>() {
            _ if alpha.is_empty() => Ok(()),
            Ok(a) if a.is_finite() && a > 0.0 => Ok(()),
            _ => Err(tf("forced.alpha.invalid", &[("value", &alpha)])),
        }
    }

    /// Appended to the system prompt after the mission.
    pub fn brief(&self) -> String {
        let operation = match self.operation {
            Operation::Auto => "celle du protocole (project_out)",
            Operation::ProjectOut => "project_out",
            Operation::Add => "add (décalage additif W += α·v, pas une ablation)",
        };
        let layers = match self.layers.trim() {
            "" => "celles que justifie la sonde",
            l => l,
        };
        let alpha = match self.alpha.trim() {
            "" => "balayage du protocole",
            a => a,
        };
        format!(
            "Réglages imposés par l'utilisateur (mode avancé CLI), prioritaires sur tes choix : \
             operation = {operation} ; projections = {} ; layers = {layers} ; alpha = {alpha}. \
             Écris-les tels quels dans chaque opération de surgeon_apply, sans les élargir ni \
             les remplacer ; l'extraction, la sonde et la validation du protocole restent dues.",
            self.tensors.projections()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_settings_are_checked_then_written_verbatim_for_the_agent() {
        crate::ui::i18n::set_lang(crate::settings::Lang::Fr);
        let mut forced = Forced::default();
        assert_eq!(forced.check(), Ok(()));
        let brief = forced.brief();
        assert!(brief.contains("project_out") && brief.contains(r#"["o_proj", "down_proj"]"#));
        assert!(brief.contains("sonde") && brief.contains("balayage"));
        forced = Forced {
            operation: Operation::Add,
            tensors: Tensors::DownProj,
            layers: "12-15".into(),
            alpha: "0.6".into(),
        };
        assert_eq!(forced.check(), Ok(()));
        let brief = forced.brief();
        for needle in [
            "operation = add",
            r#"["down_proj"]"#,
            "layers = 12-15",
            "alpha = 0.6",
        ] {
            assert!(brief.contains(needle), "{needle} in {brief}");
        }
        forced.layers = "12-".into();
        assert!(forced.check().unwrap_err().starts_with("Couches"));
        forced.layers = "all".into();
        for bad in ["0", "-1", "abc", "NaN", "inf"] {
            forced.alpha = bad.into();
            assert!(forced.check().is_err(), "{bad}");
        }
    }

    #[test]
    fn each_mission_names_its_method_and_keeps_the_approval_rule() {
        let advance = Mission::Advance.brief();
        assert!(advance.contains("surgeon_graph") && advance.contains("une par une"));
        for symptom in Symptom::ALL {
            let brief = Mission::Scalping(symptom).brief();
            assert!(brief.contains(symptom.lead()), "{symptom:?}");
            assert!(brief.contains(symptom.probe()) && brief.contains("builtin:"));
            assert!(brief.contains("Ne fais pas de scan complet"));
            assert!(brief.contains("approbation"));
        }
    }

    #[test]
    fn a_mission_round_trips_through_json() {
        for m in [Mission::Advance, Mission::Scalping(Symptom::Refusals)] {
            let json = serde_json::to_string(&m).unwrap();
            assert_eq!(serde_json::from_str::<Mission>(&json).unwrap(), m);
        }
        assert_eq!(
            serde_json::to_string(&Mission::Scalping(Symptom::Language)).unwrap(),
            r#"{"kind":"scalping","symptom":"language"}"#
        );
    }
}
