//! Evaluation datasets: a JSON array (JSONL as a fallback) of objects with `question` (or
//! `problem`), `answer` (or `solution`) and an optional `id`. The gold answer is read per
//! format: GSM8K takes the text after the last `####`, MATH the content of the last
//! `\boxed{...}` (nested braces handled by a parser, not a regex), raw takes the whole
//! answer. `--limit` keeps the first N entries in file order: no random sampling.

use std::collections::HashSet;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::{BenchError, Result};
use super::extract_answer::last_boxed;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema, clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum DatasetFormat {
    /// Réponse après le dernier `####`.
    Gsm8k,
    /// Contenu du dernier `\boxed{...}`.
    Math,
    /// Par entrée : `####`, sinon `\boxed{}`, sinon la réponse brute.
    #[default]
    Auto,
}

impl DatasetFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gsm8k => "gsm8k",
            Self::Math => "math",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub id: String,
    pub question: String,
    /// Gold answer as extracted from the reference (not yet normalized).
    pub gold: String,
}

fn text_field(entry: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| match entry.get(*k)? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    })
}

/// Gold answer of one reference for `format`.
pub fn gold_answer(reference: &str, format: DatasetFormat) -> Option<String> {
    let hashes = || {
        reference
            .rsplit_once("####")
            .map(|(_, rest)| rest.trim().to_owned())
    };
    let answer = match format {
        DatasetFormat::Gsm8k => hashes(),
        DatasetFormat::Math => last_boxed(reference),
        DatasetFormat::Auto => hashes()
            .or_else(|| last_boxed(reference))
            .or_else(|| Some(reference.trim().to_owned())),
    };
    answer.filter(|a| !a.trim().is_empty())
}

/// The first `limit` entries in file order.
pub fn load(path: &Path, format: DatasetFormat, limit: usize) -> Result<Vec<Item>> {
    if limit == 0 {
        return Err(BenchError::InvalidArgument("--limit doit être >= 1".into()));
    }
    let raw = std::fs::read_to_string(path).map_err(|e| BenchError::input(path, e))?;
    let content = raw.trim_start_matches('\u{feff}').trim();
    let entries: Vec<Value> = if content.starts_with('[') {
        let all: Vec<Value> = serde_json::from_str(content)
            .map_err(|e| BenchError::input(path, format!("JSON invalide : {e}")))?;
        all.into_iter().take(limit).collect()
    } else {
        content
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
            .take(limit)
            .map(|(i, l)| {
                serde_json::from_str(l).map_err(|e| {
                    BenchError::input(path, format!("JSONL invalide ligne {} : {e}", i + 1))
                })
            })
            .collect::<Result<_>>()?
    };
    if entries.is_empty() {
        return Err(BenchError::input(path, "dataset vide"));
    }
    let mut seen = HashSet::new();
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let bad = |m: &str| BenchError::input(path, format!("entrée {} : {m}", i + 1));
            if !entry.is_object() {
                return Err(bad("objet JSON attendu"));
            }
            let question = text_field(entry, &["question", "problem"])
                .filter(|q| !q.trim().is_empty())
                .ok_or_else(|| bad("champ question (ou problem) manquant ou vide"))?;
            let reference = text_field(entry, &["answer", "solution"])
                .ok_or_else(|| bad("champ answer (ou solution) manquant"))?;
            let gold = gold_answer(&reference, format).ok_or_else(|| {
                bad(&format!(
                    "réponse de référence introuvable au format {}",
                    format.as_str()
                ))
            })?;
            let id = text_field(entry, &["id"])
                .filter(|id| !id.trim().is_empty())
                .unwrap_or_else(|| format!("item-{}", i + 1));
            if !seen.insert(id.clone()) {
                return Err(bad(&format!("id « {id} » dupliqué")));
            }
            Ok(Item { id, question, gold })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.json");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn formats_limit_and_order() {
        let (_d, p) = file(
            r#"[{"question":"Q1","answer":"So 3+4=7.\n#### 7"},
               {"problem":"Q2","solution":"Thus $\\boxed{\\frac{1}{2}}$."},
               {"id":7,"question":"Q3","answer":"blue"}]"#,
        );
        let items = load(&p, DatasetFormat::Auto, 50).unwrap();
        let golds: Vec<&str> = items.iter().map(|i| i.gold.as_str()).collect();
        assert_eq!(golds, ["7", "\\frac{1}{2}", "blue"]);
        let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["item-1", "item-2", "7"]);
        assert_eq!(load(&p, DatasetFormat::Auto, 2).unwrap().len(), 2);
        assert!(
            load(&p, DatasetFormat::Gsm8k, 50).is_err(),
            "Q2 has no ####"
        );
        assert!(load(&p, DatasetFormat::Math, 1).is_err(), "Q1 has no box");
        assert_eq!(load(&p, DatasetFormat::Gsm8k, 1).unwrap()[0].gold, "7");
        assert!(load(&p, DatasetFormat::Auto, 0).is_err());
    }

    #[test]
    fn jsonl_and_malformed_inputs() {
        let (_d, p) = file("{\"question\":\"a\",\"answer\":\"#### 1\"}\n\n{\"question\":\"b\",\"answer\":\"#### 2\"}\n");
        assert_eq!(load(&p, DatasetFormat::Gsm8k, 10).unwrap().len(), 2);
        let (_d, p) = file("[{\"question\": \"a\", ");
        assert!(load(&p, DatasetFormat::Auto, 10).is_err());
        let (_d, p) = file("[{\"answer\":\"#### 1\"}]");
        let err = load(&p, DatasetFormat::Auto, 10).unwrap_err();
        assert!(err.to_string().contains("question"), "{err}");
        let (_d, p) = file(
            r#"[{"id":"a","question":"x","answer":"1"},{"id":"a","question":"y","answer":"2"}]"#,
        );
        assert!(load(&p, DatasetFormat::Auto, 10).is_err(), "duplicate id");
        let (_d, p) = file("[]");
        assert!(load(&p, DatasetFormat::Auto, 10).is_err());
    }
}
