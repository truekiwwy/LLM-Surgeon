use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::error::{EvalError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalPrompt {
    /// Optional in the file: missing ids become `prompt-<n>` (1-based).
    #[serde(default)]
    pub id: String,
    pub text: String,
    /// Case-insensitive substrings that must all appear in the output.
    #[serde(default)]
    pub expect_contains: Option<Vec<String>>,
    /// Case-insensitive substrings that must not appear in the output.
    #[serde(default)]
    pub expect_not_contains: Option<Vec<String>>,
}

/// Format detection on content: starts with `[` → JSON array, first line starts with `{`
/// → JSONL, otherwise plain text with one prompt per non-empty line. A plain-text file
/// whose first line starts with `[` or `{` is therefore read as JSON.
pub fn load_prompts(path: &Path) -> Result<Vec<EvalPrompt>> {
    let err = |message: String| EvalError::Prompts {
        path: path.to_path_buf(),
        message,
    };
    let raw = std::fs::read_to_string(path).map_err(|e| err(e.to_string()))?;
    // Notepad writes a BOM, which serde_json rejects.
    let content = raw.trim_start_matches('\u{feff}').trim();

    let mut prompts: Vec<EvalPrompt> = if content.starts_with('[') {
        serde_json::from_str(content).map_err(|e| err(format!("JSON invalide : {e}")))?
    } else if content.starts_with('{') {
        content
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
            .map(|(i, l)| {
                serde_json::from_str(l)
                    .map_err(|e| err(format!("JSONL invalide ligne {} : {e}", i + 1)))
            })
            .collect::<Result<_>>()?
    } else {
        content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| EvalPrompt {
                id: String::new(),
                text: l.to_owned(),
                expect_contains: None,
                expect_not_contains: None,
            })
            .collect()
    };

    if prompts.is_empty() {
        return Err(err("aucun prompt".into()));
    }
    let mut seen = HashSet::new();
    for (i, p) in prompts.iter_mut().enumerate() {
        if p.id.trim().is_empty() {
            p.id = format!("prompt-{}", i + 1);
        }
        if p.text.trim().is_empty() {
            return Err(err(format!("prompt « {} » : texte vide", p.id)));
        }
        if !seen.insert(p.id.clone()) {
            return Err(err(format!("id « {} » dupliqué", p.id)));
        }
    }
    Ok(prompts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(content: &str) -> Result<Vec<EvalPrompt>> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts");
        std::fs::write(&path, content).unwrap();
        load_prompts(&path)
    }

    fn message(r: Result<Vec<EvalPrompt>>) -> String {
        match r {
            Err(EvalError::Prompts { message, .. }) => message,
            other => panic!("expected Prompts error, got {other:?}"),
        }
    }

    #[test]
    fn json_array() {
        let p = load(
            r#"[{"id":"capital","text":"Capitale de la France ?","expect_contains":["Paris"]},
                {"text":"Bonjour","expect_not_contains":["Wait"]}]"#,
        )
        .unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].id, "capital");
        assert_eq!(p[0].expect_contains, Some(vec!["Paris".to_owned()]));
        assert_eq!(p[1].id, "prompt-2");
        assert_eq!(p[1].expect_not_contains, Some(vec!["Wait".to_owned()]));
    }

    #[test]
    fn jsonl_with_bom_and_crlf() {
        let p =
            load("\u{feff}{\"id\":\"a\",\"text\":\"un\"}\r\n\r\n{\"text\":\"deux\"}\r\n").unwrap();
        assert_eq!(
            p.iter()
                .map(|p| (p.id.as_str(), p.text.as_str()))
                .collect::<Vec<_>>(),
            [("a", "un"), ("prompt-2", "deux")]
        );
        assert!(message(load("{\"text\":\"ok\"}\n{\"text\":\n")).contains("ligne 2"));
    }

    #[test]
    fn plain_text_one_prompt_per_line() {
        let p = load("Quelle heure est-il ?\n\n   \nRésume ce texte : é🦀\n").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[1].text, "Résume ce texte : é🦀");
        assert_eq!(p[1].id, "prompt-2");
    }

    #[test]
    fn invalid_files_are_clear_errors() {
        assert_eq!(message(load("")), "aucun prompt");
        assert_eq!(message(load("\u{feff}  \n\t\n")), "aucun prompt");
        assert_eq!(message(load("[]")), "aucun prompt");
        assert!(message(load(r#"[{"id":"a","text": }]"#)).starts_with("JSON invalide"));
        assert!(message(load(r#"[{"txt":"typo"}]"#)).contains("txt"));
        assert!(
            message(load(r#"[{"id":"a","text":"x"},{"id":"a","text":"y"}]"#)).contains("dupliqué")
        );
        assert!(message(load(r#"[{"text":"  "}]"#)).contains("texte vide"));
        let missing = load_prompts(Path::new("introuvable/prompts.json")).unwrap_err();
        assert!(matches!(missing, EvalError::Prompts { .. }));
    }
}
