//! What a session is for, chosen when it starts and kept with its conversation:
//! `Advance` (the agent scans the whole model and finds the problems itself) or `Scalping`
//! (the user names a precise symptom and the agent goes straight to the likely tensors).
//! The brief goes to the agent with the system prompt; the approval policy is unchanged:
//! the scan only reads, every write is a card the user approves.

use serde::{Deserialize, Serialize};

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
            Symptom::Repetition | Symptom::Language | Symptom::Other => {
                "builtin:general (surgeon_bench mode eval, en lisant les générations)"
            }
        }
    }

    /// Where the agent looks first, for the brief.
    fn lead(self) -> &'static str {
        match self {
            Symptom::Refusals => {
                "refus excessifs : direction de refus dans le flux résiduel des couches du \
                 milieu, projections o_proj et down_proj ; mesure avec surgeon_bench (refusal)"
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
            Symptom::Other => "autre : déduis les composants probables de la description",
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
                 4. Ensuite seulement, propose les corrections une par une, chacune soumise \
                 à l'approbation de l'utilisateur, et mesure avant et après."
                .into(),
            Mission::Scalping(symptom) => format!(
                "Mission : scalping. L'utilisateur a signalé un problème précis ; sa \
                 description est dans son message. Symptôme : {}. Jeu de mesure : {}.\n\
                 Ne fais pas de scan complet : va directement aux tenseurs les plus \
                 susceptibles de l'expliquer, mesure-les (surgeon_inspect, surgeon_graph \
                 limité à ces couches, surgeon_bench en lecture), confirme ou infirme \
                 l'hypothèse avec record_finding, montre la cible avec show_in_3d, puis \
                 propose la correction la plus ciblée possible, soumise à approbation, et \
                 mesure avant et après.",
                symptom.lead(),
                symptom.probe()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
