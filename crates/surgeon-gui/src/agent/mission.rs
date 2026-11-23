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

/// How `surgeon_compose` merges the directions to remove, forced from the advanced panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Merge {
    /// The protocol's choice (orthogonal-preserve when a capability is kept).
    #[default]
    Auto,
    Linear,
    Orthogonal,
    Ties,
    Dare,
}

impl Merge {
    /// The `mode` of `surgeon_compose`; `None`: the agent chooses.
    pub fn mode(self) -> Option<&'static str> {
        match self {
            Merge::Auto => None,
            Merge::Linear => Some("standard"),
            Merge::Orthogonal => Some("orthogonal_preserve"),
            Merge::Ties => Some("ties"),
            Merge::Dare => Some("dare"),
        }
    }
}

/// What the user forces from the advanced (CLI) panel, kept with the conversation like the
/// mission: the parts of the patch the agent must not choose. An empty `layers` or `alpha`
/// stays the protocol's (the probe, the alpha sweep), an empty merge setting the engine's
/// default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Forced {
    pub operation: Operation,
    pub tensors: Tensors,
    /// `all`, `12-15` or `3,5`: the `layers` of a patch operation.
    pub layers: String,
    pub alpha: String,
    /// Absent from the conversations saved before it: the protocol's choice.
    #[serde(default)]
    pub merge: Merge,
    /// TIES: share of each vector kept, in ]0, 1].
    #[serde(default)]
    pub density: String,
    /// DARE: probability of dropping a coordinate, in [0, 0.95].
    #[serde(default)]
    pub drop_rate: String,
    /// DARE: seed of the masks.
    #[serde(default)]
    pub seed: String,
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
            _ if alpha.is_empty() => {}
            Ok(a) if a.is_finite() && a > 0.0 => {}
            _ => return Err(tf("forced.alpha.invalid", &[("value", &alpha)])),
        }
        // Only the settings of the merge on screen are sent: the others are not checked.
        // `Some(text)`: a value outside `ok` (empty keeps the engine's default).
        let bad = |text: &str, ok: fn(f64) -> bool| {
            let text = text.trim();
            let fine = text.is_empty() || text.parse::<f64>().is_ok_and(ok);
            (!fine).then(|| text.to_owned())
        };
        match self.merge {
            Merge::Ties => match bad(&self.density, |v| v > 0.0 && v <= 1.0) {
                Some(v) => Err(tf("forced.density.invalid", &[("value", &v)])),
                None => Ok(()),
            },
            Merge::Dare => {
                if let Some(v) = bad(&self.drop_rate, |v| (0.0..=0.95).contains(&v)) {
                    return Err(tf("forced.drop_rate.invalid", &[("value", &v)]));
                }
                let seed = self.seed.trim();
                match seed.is_empty() || seed.parse::<u64>().is_ok() {
                    true => Ok(()),
                    false => Err(tf("forced.seed.invalid", &[("value", &seed)])),
                }
            }
            _ => Ok(()),
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
        let mut brief = format!(
            "Réglages imposés par l'utilisateur (mode avancé CLI), prioritaires sur tes choix : \
             operation = {operation} ; projections = {} ; layers = {layers} ; alpha = {alpha}. \
             Écris-les tels quels dans chaque opération de surgeon_apply, sans les élargir ni \
             les remplacer ; l'extraction, la sonde et la validation du protocole restent dues.",
            self.tensors.projections()
        );
        if let Some(mode) = self.merge.mode() {
            let or_default = |v: &str| match v.trim() {
                "" => "défaut du moteur".to_owned(),
                v => v.to_owned(),
            };
            let settings = match self.merge {
                Merge::Ties => format!(" ; density = {}", or_default(&self.density)),
                Merge::Dare => format!(
                    " ; drop_rate = {} ; seed = {}",
                    or_default(&self.drop_rate),
                    or_default(&self.seed)
                ),
                _ => String::new(),
            };
            brief.push_str(&format!(
                " Fusion imposée : chaque surgeon_compose prend mode = {mode}{settings}, sans \
                 changer de méthode."
            ));
        }
        brief
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
        assert!(!brief.contains("Fusion"), "no merge forced by default");
        forced = Forced {
            operation: Operation::Add,
            tensors: Tensors::DownProj,
            layers: "12-15".into(),
            alpha: "0.6".into(),
            ..Forced::default()
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
    fn a_forced_merge_names_the_compose_mode_and_only_its_own_settings() {
        crate::ui::i18n::set_lang(crate::settings::Lang::Fr);
        let mut forced = Forced {
            merge: Merge::Ties,
            density: "0.3".into(),
            drop_rate: "nope".into(),
            ..Forced::default()
        };
        // The DARE field is not the one on screen: it does not block a TIES merge.
        assert_eq!(forced.check(), Ok(()));
        let brief = forced.brief();
        assert!(brief.contains("mode = ties ; density = 0.3"), "{brief}");
        assert!(!brief.contains("drop_rate"), "{brief}");
        for bad in ["0", "1.5", "x"] {
            forced.density = bad.into();
            assert!(forced.check().unwrap_err().starts_with("Densité"), "{bad}");
        }

        forced = Forced {
            merge: Merge::Dare,
            ..Forced::default()
        };
        let brief = forced.brief();
        assert!(
            brief.contains("mode = dare ; drop_rate = défaut du moteur ; seed = défaut du moteur"),
            "{brief}"
        );
        forced.drop_rate = "0.96".into();
        assert!(forced.check().is_err());
        forced.drop_rate = "0.5".into();
        forced.seed = "-3".into();
        assert!(forced.check().is_err());
        forced.seed = "42".into();
        assert_eq!(forced.check(), Ok(()));
        let modes = [Merge::Linear, Merge::Orthogonal].map(|m| m.mode());
        assert_eq!(modes, [Some("standard"), Some("orthogonal_preserve")]);
        assert_eq!(Merge::Auto.mode(), None);
        // A conversation saved before the merge settings still reads.
        let old = r#"{"operation": "add", "tensors": "both", "layers": "", "alpha": ""}"#;
        let read: Forced = serde_json::from_str(old).unwrap();
        assert_eq!(read.merge, Merge::Auto);
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
