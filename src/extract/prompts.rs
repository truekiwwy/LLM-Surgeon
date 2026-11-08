//! Prompt files, chat template and tokenization.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::error::{ExtractError, Result};

/// How each prompt is wrapped before tokenization.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum ChatTemplate {
    /// Raw text.
    #[default]
    #[value(name = "none")]
    #[serde(rename = "none")]
    NoTemplate,
    /// `<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n` (Qwen chat format).
    #[value(name = "chatml")]
    Chatml,
}

impl ChatTemplate {
    pub fn as_str(self) -> &'static str {
        match self {
            ChatTemplate::NoTemplate => "none",
            ChatTemplate::Chatml => "chatml",
        }
    }

    pub fn apply(self, prompt: &str) -> String {
        match self {
            ChatTemplate::NoTemplate => prompt.to_owned(),
            ChatTemplate::Chatml => {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
        }
    }
}

/// A JSON array of strings, else whatever `eval::load_prompts` reads (array of objects,
/// JSONL, plain text), keeping only the texts.
pub fn load_prompt_texts(path: &Path) -> std::result::Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{} : {e}", path.display()))?;
    let content = raw.trim_start_matches('\u{feff}').trim();
    if content.starts_with('[') {
        if let Ok(texts) = serde_json::from_str::<Vec<String>>(content) {
            return check_texts(texts);
        }
    }
    let prompts = crate::eval::load_prompts(path).map_err(|e| e.to_string())?;
    check_texts(prompts.into_iter().map(|p| p.text).collect())
}

/// At least one prompt, none empty.
pub fn check_texts(texts: Vec<String>) -> std::result::Result<Vec<String>, String> {
    if texts.is_empty() {
        return Err("aucun prompt".into());
    }
    match texts.iter().position(|t| t.trim().is_empty()) {
        Some(i) => Err(format!("prompt {i} vide")),
        None => Ok(texts),
    }
}

pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&path)
        .map_err(|e| ExtractError::Tokenizer(format!("{} : {e}", path.display())))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenized {
    pub ids: Vec<u32>,
    pub truncated: bool,
}

/// Right truncation to `max_tokens`. Batch of one: no padding, so no padding token needed.
pub fn tokenize(tokenizer: &Tokenizer, text: &str, max_tokens: usize) -> Result<Tokenized> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| ExtractError::Tokenizer(e.to_string()))?;
    let mut ids = encoding.get_ids().to_vec();
    let truncated = ids.len() > max_tokens;
    ids.truncate(max_tokens);
    Ok(Tokenized { ids, truncated })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::AddedToken;

    fn write(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn every_prompt_file_format() {
        let (_d, p) = write(r#"["un", "deux"]"#);
        assert_eq!(load_prompt_texts(&p).unwrap(), ["un", "deux"]);
        let (_d, p) = write(r#"[{"text": "un"}, {"id": "b", "text": "deux"}]"#);
        assert_eq!(load_prompt_texts(&p).unwrap(), ["un", "deux"]);
        let (_d, p) = write("{\"text\":\"un\"}\n{\"text\":\"deux\"}\n");
        assert_eq!(load_prompt_texts(&p).unwrap(), ["un", "deux"]);
        let (_d, p) = write("un\n\ndeux\n");
        assert_eq!(load_prompt_texts(&p).unwrap(), ["un", "deux"]);
        let (_d, p) = write("");
        assert!(load_prompt_texts(&p).unwrap_err().contains("aucun prompt"));
        let (_d, p) = write(r#"["un", "#);
        assert!(load_prompt_texts(&p).unwrap_err().contains("JSON invalide"));
        let (_d, p) = write(r#"["un", "  "]"#);
        assert!(load_prompt_texts(&p).unwrap_err().contains("vide"));
    }

    fn word_tokenizer() -> Tokenizer {
        let vocab = [
            "[UNK]",
            "<|im_start|>",
            "<|im_end|>",
            "user",
            "assistant",
            "bonjour",
            "le",
            "monde",
        ];
        let model = WordLevel::builder()
            .vocab(
                vocab
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (w.to_string(), i as u32))
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        tokenizer
            .add_special_tokens([
                AddedToken::from("<|im_start|>", true),
                AddedToken::from("<|im_end|>", true),
            ])
            .unwrap();
        tokenizer
    }

    #[test]
    fn chatml_keeps_special_tokens_whole_and_truncation_is_reported() {
        let tok = word_tokenizer();
        let wrapped = ChatTemplate::Chatml.apply("bonjour le monde");
        let t = tokenize(&tok, &wrapped, 512).unwrap();
        assert_eq!(
            t.ids,
            [1, 3, 5, 6, 7, 2, 1, 4],
            "<|im_start|> user ... <|im_end|> <|im_start|> assistant"
        );
        assert!(!t.truncated);
        let cut = tokenize(&tok, &wrapped, 3).unwrap();
        assert_eq!(cut.ids, [1, 3, 5]);
        assert!(cut.truncated);
        assert_eq!(ChatTemplate::NoTemplate.apply("x"), "x");
    }

    /// The Qwen2 pre-tokenizer (Split with a lookahead regex, then ByteLevel) must work
    /// with the pure-Rust regex engine (fancy-regex): `\s+(?!\S)` keeps one space apart.
    #[test]
    fn qwen2_lookahead_regex_runs_on_fancy_regex() {
        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "post_processor": null, "decoder": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]},
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "Hello": 1, "Ġ": 2, "Ġworld": 3}, "unk_token": "[UNK]"}
        });
        let tok: Tokenizer = json.to_string().parse().unwrap();
        let encoding = tok.encode("Hello  world", false).unwrap();
        assert_eq!(encoding.get_tokens(), ["Hello", "Ġ", "Ġworld"]);
        assert_eq!(encoding.get_ids(), [1, 2, 3]);
    }
}
