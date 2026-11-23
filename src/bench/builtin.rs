//! Probe sets embedded in the binary, named `builtin:<set>` wherever a dataset or a prompts
//! file is expected, so that a measure needs no file of the user's.
//!
//! - `math`, `logic`, `general`: questions with a numeric gold answer, for `bench eval`
//!   (exact match reads the final number reliably);
//! - `refusals`: legitimate requests that over-cautious models refuse, for `bench refusal`
//!   (and any set's questions as prompts for `refusal` or `dump`).

use std::path::Path;

pub const PREFIX: &str = "builtin:";

/// Name, content, whether it holds gold answers (usable by `eval`).
const SETS: [(&str, &str, bool); 4] = [
    ("math", include_str!("builtin/math.json"), true),
    ("logic", include_str!("builtin/logic.json"), true),
    ("general", include_str!("builtin/general.json"), true),
    ("refusals", include_str!("builtin/refusals.json"), false),
];

const TEXTS: [(&str, &str); 1] = [("wikitext", include_str!("builtin/wikitext.txt"))];

pub fn text(path: &Path) -> Option<Result<&'static str, String>> {
    let name = path.to_str()?.strip_prefix(PREFIX)?;
    Some(
        TEXTS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, text)| *text)
            .ok_or_else(|| {
                format!("texte intégré inconnu « {name} » ; disponible : {PREFIX}wikitext")
            }),
    )
}

/// The set names, for the tool descriptions and the agent prompts.
pub fn names() -> String {
    SETS.iter()
        .map(|(name, ..)| format!("{PREFIX}{name}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether `path` names a built-in set (known or not): it is never a file to sandbox.
pub fn is_builtin(path: &Path) -> bool {
    path.to_str().is_some_and(|p| p.starts_with(PREFIX))
}

fn set(path: &Path) -> Option<Result<(&'static str, &'static str, bool), String>> {
    let name = path.to_str()?.strip_prefix(PREFIX)?;
    Some(
        SETS.iter()
            .find(|(n, ..)| *n == name)
            .copied()
            .ok_or_else(|| format!("jeu intégré inconnu « {name} » ; disponibles : {}", names())),
    )
}

/// `Some` for a `builtin:` path: the JSON of a set with gold answers, or why it cannot serve
/// `eval`.
pub fn dataset(path: &Path) -> Option<Result<&'static str, String>> {
    set(path).map(|found| {
        let (name, json, answers) = found?;
        match answers {
            true => Ok(json),
            false => Err(format!(
                "{PREFIX}{name} n'a pas de réponses de référence : il se mesure avec le mode refusal (prompts)"
            )),
        }
    })
}

/// `Some` for a `builtin:` path: the questions of the set, as prompts.
pub fn prompts(path: &Path) -> Option<Result<Vec<String>, String>> {
    set(path).map(|found| {
        let (_, json, _) = found?;
        let entries: Vec<serde_json::Value> =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        Ok(entries
            .iter()
            .filter_map(|e| e.get("question")?.as_str().map(str::to_owned))
            .collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::dataset::{load, DatasetFormat};

    #[test]
    fn every_set_loads_with_15_to_20_items() {
        for (name, _, answers) in SETS {
            let path = format!("{PREFIX}{name}");
            let path = Path::new(&path);
            assert!(is_builtin(path));
            let questions = prompts(path).unwrap().unwrap();
            assert!(
                (15..=20).contains(&questions.len()),
                "{name}: {}",
                questions.len()
            );
            if answers {
                let items = load(path, DatasetFormat::Auto, 100).unwrap();
                assert_eq!(items.len(), questions.len(), "{name}");
                assert!(
                    items.iter().all(|i| i.gold.parse::<f64>().is_ok()),
                    "{name}: numeric gold answers"
                );
            }
        }
    }

    #[test]
    fn refusals_is_for_the_refusal_mode_and_unknown_sets_are_named() {
        let err = load(Path::new("builtin:refusals"), DatasetFormat::Auto, 10).unwrap_err();
        assert!(err.to_string().contains("refusal"), "{err}");
        let err = dataset(Path::new("builtin:poetry")).unwrap().unwrap_err();
        assert!(err.contains("builtin:math"), "{err}");
        assert!(dataset(Path::new("data/gsm8k.jsonl")).is_none());
    }
}
