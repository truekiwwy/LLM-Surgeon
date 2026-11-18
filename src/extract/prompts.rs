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
    /// The format the model declares (`tokenizer_config.json`, `chat_template.jinja`, GGUF
    /// metadata), its default system message included.
    #[default]
    #[value(name = "auto")]
    Auto,
    /// Raw text.
    #[value(name = "none")]
    #[serde(rename = "none")]
    NoTemplate,
    /// `<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n` (Qwen chat format).
    #[value(name = "chatml")]
    Chatml,
    /// `<|start_header_id|>user<|end_header_id|>` turns (Llama 3).
    #[value(name = "llama3")]
    Llama3,
    /// `[INST] {p} [/INST]` (Llama 2, Mistral).
    #[value(name = "inst")]
    Inst,
    /// `<start_of_turn>user` turns (Gemma).
    #[value(name = "gemma")]
    Gemma,
}

/// The families `auto` recognizes. There is no Jinja engine here: the declared template only
/// selects the family, whose turns are written in [`PromptFormat::apply`].
const FAMILIES: [ChatTemplate; 4] = [
    ChatTemplate::Chatml,
    ChatTemplate::Llama3,
    ChatTemplate::Gemma,
    ChatTemplate::Inst,
];

impl ChatTemplate {
    pub fn as_str(self) -> &'static str {
        match self {
            ChatTemplate::Auto => "auto",
            ChatTemplate::NoTemplate => "none",
            ChatTemplate::Chatml => "chatml",
            ChatTemplate::Llama3 => "llama3",
            ChatTemplate::Inst => "inst",
            ChatTemplate::Gemma => "gemma",
        }
    }

    /// The text that opens a turn of this family.
    fn marker(self) -> Option<&'static str> {
        match self {
            ChatTemplate::Auto | ChatTemplate::NoTemplate => None,
            ChatTemplate::Chatml => Some("<|im_start|>"),
            ChatTemplate::Llama3 => Some("<|start_header_id|>"),
            ChatTemplate::Inst => Some("[INST]"),
            ChatTemplate::Gemma => Some("<start_of_turn>"),
        }
    }

    /// This choice without looking at a model: `auto` is then raw text.
    pub fn format(self) -> PromptFormat {
        PromptFormat {
            template: match self {
                ChatTemplate::Auto => ChatTemplate::NoTemplate,
                other => other,
            },
            system: None,
        }
    }

    /// This choice for `model` (a folder, a weights file or a GGUF). `fallback` is what
    /// `auto` gives when the model declares no format the engine knows.
    pub fn resolve(self, model: &Path, fallback: ChatTemplate) -> PromptFormat {
        match self {
            ChatTemplate::Auto => detect_format(model).unwrap_or_else(|| fallback.format()),
            other => other.format(),
        }
    }

    pub fn apply(self, prompt: &str) -> String {
        self.format().apply(prompt)
    }
}

/// A resolved wrapping: never `auto`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptFormat {
    pub template: ChatTemplate,
    /// The default system message of the model's own template (ChatML only).
    pub system: Option<String>,
}

impl PromptFormat {
    /// The family name for a report; None for raw text.
    pub fn family(&self) -> Option<&'static str> {
        self.template.marker().map(|_| self.template.as_str())
    }

    /// The begin-of-text token is left to the tokenizer's post-processor. A prompt already
    /// holding this family's turns (a rendered prompt) is kept as is, never wrapped twice.
    pub fn apply(&self, prompt: &str) -> String {
        if self.template.marker().is_none_or(|m| prompt.contains(m)) {
            return prompt.to_owned();
        }
        match self.template {
            ChatTemplate::Chatml => {
                let system = self.system.as_deref().map_or(String::new(), |s| {
                    format!("<|im_start|>system\n{s}<|im_end|>\n")
                });
                format!("{system}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
            ChatTemplate::Llama3 => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            ChatTemplate::Inst => format!("[INST] {prompt} [/INST]"),
            ChatTemplate::Gemma => {
                format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
            }
            ChatTemplate::Auto | ChatTemplate::NoTemplate => prompt.to_owned(),
        }
    }
}

/// The folder of a model given by one of its `.safetensors` files; any other path as is.
pub fn model_folder(model: &Path) -> &Path {
    let weights = model
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    match model.parent() {
        Some(dir) if weights && model.is_file() && dir.join("config.json").is_file() => dir,
        _ => model,
    }
}

/// The family of the declared template, else of the special tokens `tokenizer_config.json`
/// lists (an older chat model without a template).
fn detect_format(model: &Path) -> Option<PromptFormat> {
    let declared = declared_template(model);
    let searched = declared.clone().or_else(|| {
        std::fs::read_to_string(crate::eval::model_dir(model).join("tokenizer_config.json")).ok()
    })?;
    let template = FAMILIES
        .into_iter()
        .find(|f| f.marker().is_some_and(|m| searched.contains(m)))?;
    let system = declared
        .as_deref()
        .filter(|_| template == ChatTemplate::Chatml)
        .and_then(default_system);
    Some(PromptFormat { template, system })
}

fn declared_template(model: &Path) -> Option<String> {
    if crate::eval::is_gguf(model) {
        return gguf_template(model);
    }
    let dir = crate::eval::model_dir(model);
    let config: Option<serde_json::Value> =
        std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|text| serde_json::from_str(text.trim_start_matches('\u{feff}')).ok());
    config
        .and_then(|c| match &c["chat_template"] {
            serde_json::Value::String(text) => Some(text.clone()),
            // Named templates: `default`, else the first one.
            serde_json::Value::Array(list) => list
                .iter()
                .find(|t| t["name"] == "default")
                .or(list.first())
                .and_then(|t| t["template"].as_str())
                .map(str::to_owned),
            _ => None,
        })
        .or_else(|| std::fs::read_to_string(dir.join("chat_template.jinja")).ok())
}

fn gguf_template(model: &Path) -> Option<String> {
    let file = std::fs::File::open(model).ok()?;
    let bytes = crate::kernel::map_read(&file).ok()?;
    let gguf = crate::patch::gguf::parse(&bytes).ok()?;
    let text = gguf.meta("tokenizer.chat_template")?.as_str()?;
    Some(text.to_owned())
}

/// The literal default system message of a ChatML Jinja template, as Qwen writes it:
/// `'<|im_start|>system\nYou are ...<|im_end|>\n'`. The other occurrences of the opening
/// concatenate `messages[0]` and hold quotes.
fn default_system(template: &str) -> Option<String> {
    const OPEN: &str = "<|im_start|>system";
    template.match_indices(OPEN).find_map(|(at, _)| {
        let rest = &template[at + OPEN.len()..];
        let rest = rest.strip_prefix("\\n").or(rest.strip_prefix('\n'))?;
        let (text, _) = rest.split_once("<|im_end|>")?;
        let literal = !text.is_empty() && !text.contains(['\'', '"', '{', '}']);
        literal.then(|| text.replace("\\n", "\n"))
    })
}

/// A JSON array of strings, else whatever `eval::load_prompts` reads (array of objects,
/// JSONL, plain text), keeping only the texts.
pub fn load_prompt_texts(path: &Path) -> std::result::Result<Vec<String>, String> {
    if let Some(set) = crate::bench::builtin::prompts(path) {
        return check_texts(set?);
    }
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

    const QWEN_TEMPLATE: &str = r"{%- if tools %}{{- '<|im_start|>system\n' }}{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{%- else %}{{- '<|im_start|>system\nYou are Qwen, a helpful assistant.<|im_end|>\n' }}{%- endif %}{%- endif %}";

    fn model_with(config: serde_json::Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();
        dir
    }

    #[test]
    fn auto_follows_the_template_the_model_declares() {
        let none = ChatTemplate::NoTemplate;
        let qwen = model_with(serde_json::json!({ "chat_template": QWEN_TEMPLATE }));
        let format = ChatTemplate::Auto.resolve(qwen.path(), none);
        assert_eq!(format.template, ChatTemplate::Chatml);
        assert_eq!(
            format.system.as_deref(),
            Some("You are Qwen, a helpful assistant.")
        );
        assert_eq!(
            format.apply("Hi"),
            "<|im_start|>system\nYou are Qwen, a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        // An explicit choice never reads the model.
        assert_eq!(ChatTemplate::Chatml.resolve(qwen.path(), none).system, None);
        assert_eq!(
            none.resolve(qwen.path(), ChatTemplate::Chatml).apply("Hi"),
            "Hi"
        );
        // Through one of the model's files too.
        let weights = qwen.path().join("model.safetensors");
        assert_eq!(ChatTemplate::Auto.resolve(&weights, none), format);

        for (text, family, wrapped) in [
            (
                "<|start_header_id|>",
                ChatTemplate::Llama3,
                "<|start_header_id|>user",
            ),
            (
                "<start_of_turn>",
                ChatTemplate::Gemma,
                "<start_of_turn>user\nHi",
            ),
            ("[INST]", ChatTemplate::Inst, "[INST] Hi [/INST]"),
        ] {
            let dir = model_with(serde_json::json!({ "chat_template": [
                {"name": "tool_use", "template": "other"},
                {"name": "default", "template": text},
            ]}));
            let format = ChatTemplate::Auto.resolve(dir.path(), none);
            assert_eq!((format.template, format.system.clone()), (family, None));
            assert!(format.apply("Hi").starts_with(wrapped), "{family:?}");
        }
    }

    #[test]
    fn auto_uses_the_fallback_when_the_model_declares_nothing() {
        let bare = tempfile::tempdir().unwrap();
        let (none, chatml) = (ChatTemplate::NoTemplate, ChatTemplate::Chatml);
        assert_eq!(ChatTemplate::Auto.resolve(bare.path(), none), none.format());
        assert_eq!(
            ChatTemplate::Auto.resolve(bare.path(), chatml),
            chatml.format()
        );
        // No template, but the special tokens of a chat model.
        let tokens =
            model_with(serde_json::json!({ "additional_special_tokens": ["<|im_start|>"] }));
        assert_eq!(
            ChatTemplate::Auto.resolve(tokens.path(), none),
            chatml.format()
        );
        // A template file instead of the JSON key.
        std::fs::write(bare.path().join("chat_template.jinja"), "<start_of_turn>").unwrap();
        assert_eq!(
            ChatTemplate::Auto.resolve(bare.path(), none).template,
            ChatTemplate::Gemma
        );
    }

    #[test]
    fn a_rendered_prompt_is_not_wrapped_twice() {
        let rendered = ChatTemplate::Chatml.apply("Hi");
        assert_eq!(ChatTemplate::Chatml.apply(&rendered), rendered);
    }

    #[test]
    fn a_weights_file_stands_for_its_model_folder() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("model.safetensors");
        std::fs::write(&weights, b"x").unwrap();
        assert_eq!(model_folder(&weights), weights, "no config.json next to it");
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        assert_eq!(model_folder(&weights), dir.path());
        assert_eq!(model_folder(dir.path()), dir.path());
        let gguf = dir.path().join("m.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert_eq!(model_folder(&gguf), gguf, "a GGUF is a model by itself");
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
