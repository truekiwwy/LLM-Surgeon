//! One-click import into Ollama: a `Modelfile` in the project folder that points at the
//! project's artifact (the newest `.gguf`, else the working copy's SafeTensors folder), with
//! the inference parameters the model ships (`generation_config.json`, its context length
//! and the end-of-turn markers of its chat template), and the `ollama create` command.
//!
//! No `TEMPLATE` is written: Ollama picks the chat format from the template the GGUF or the
//! tokenizer carries, and a hand-written Go template that drifts from it breaks the chat.

use std::path::{Path, PathBuf};

use serde_json::Value;
use weight_surgeon::graph::model::plain_path;

pub const MODELFILE: &str = "Modelfile";

/// Ollama's own default is smaller; past this a long-context model eats the RAM of the
/// machine on its first load, so the full length is left to the user.
const MAX_CTX: u64 = 8192;

/// End-of-turn tokens of the common chat formats (ChatML, Llama 3, Gemma, Phi-3): each one
/// the model's template uses becomes a `stop`.
const STOPS: [&str; 5] = [
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
    "<|end|>",
    "<|endoftext|>",
];

/// `generation_config.json` key, then the Ollama parameter it becomes.
const SAMPLING: [(&str, &str); 4] = [
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("repetition_penalty", "repeat_penalty"),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Artifact {
    Gguf(PathBuf),
    /// A model folder (`config.json` and its `.safetensors`), converted by `ollama create`.
    SafeTensors(PathBuf),
}

impl Artifact {
    pub fn path(&self) -> &Path {
        match self {
            Self::Gguf(path) | Self::SafeTensors(path) => path,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub artifact: Artifact,
    pub modelfile: PathBuf,
    pub name: String,
    pub command: String,
}

/// The newest `.gguf` under the project (an `export_gguf` of `surgeon_eval`, or the working
/// copy of a GGUF model), else `copy` when it is a SafeTensors model folder.
pub fn find_artifact(project: &Path, copy: Option<&Path>) -> Option<Artifact> {
    let mut found = Vec::new();
    ggufs(project, 4, &mut found);
    let newest = found
        .into_iter()
        .max_by_key(|path| std::fs::metadata(path).and_then(|m| m.modified()).ok());
    match newest {
        Some(gguf) => Some(Artifact::Gguf(gguf)),
        None => copy
            .filter(|dir| dir.join("config.json").is_file())
            .map(|dir| Artifact::SafeTensors(dir.to_path_buf())),
    }
}

fn ggufs(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for path in entries.flatten().map(|entry| entry.path()) {
        if path.is_dir() && depth > 0 {
            ggufs(&path, depth - 1, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

/// An Ollama model name: lowercase ASCII letters, digits, `.`, `_` and `-`.
pub fn model_name(project: &str) -> String {
    let mapped: String = project
        .chars()
        .map(|c| match c.is_ascii_alphanumeric() || "._".contains(c) {
            true => c.to_ascii_lowercase(),
            false => '-',
        })
        .collect();
    let name = mapped
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    match name.is_empty() {
        true => "surgeon-model".into(),
        false => name,
    }
}

/// A path as a `Modelfile` value: verbatim prefix dropped, forward slashes (which Windows
/// takes), quoted when it holds a space.
fn modelfile_path(path: &Path) -> String {
    let text = plain_path(path.to_path_buf()).display().to_string();
    let text = match cfg!(windows) {
        true => text.replace('\\', "/"),
        false => text,
    };
    match text.contains(char::is_whitespace) {
        true => format!("\"{text}\""),
        false => text,
    }
}

fn read_json(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null)
}

/// The `Modelfile` of `artifact`, its parameters read from the model folder `config`.
pub fn modelfile(artifact: &Artifact, config: Option<&Path>) -> String {
    let mut lines = vec![format!("FROM {}", modelfile_path(artifact.path()))];
    if let Some(dir) = config {
        let generation = read_json(&dir.join("generation_config.json"));
        for (key, parameter) in SAMPLING {
            if let Some(value) = generation.get(key).filter(|v| v.is_number()) {
                lines.push(format!("PARAMETER {parameter} {value}"));
            }
        }
        let context = read_json(&dir.join("config.json"))["max_position_embeddings"].as_u64();
        if let Some(context) = context {
            lines.push(format!("PARAMETER num_ctx {}", context.min(MAX_CTX)));
        }
        let tokenizer = read_json(&dir.join("tokenizer_config.json"));
        let template = tokenizer["chat_template"].as_str().unwrap_or_default();
        for stop in STOPS.iter().filter(|stop| template.contains(*stop)) {
            lines.push(format!("PARAMETER stop \"{stop}\""));
        }
    }
    lines.join("\n") + "\n"
}

/// Writes `<project>/Modelfile` for `artifact`, else the project's artifact; `None` when
/// there is none yet. Parameters come from the working copy, else the origin, whichever is a
/// model folder.
pub fn export(
    project: &Path,
    project_name: &str,
    artifact: Option<Artifact>,
    copy: Option<&Path>,
    origin: Option<&Path>,
) -> Result<Option<Export>, String> {
    let Some(artifact) = artifact.or_else(|| find_artifact(project, copy)) else {
        return Ok(None);
    };
    let config = [copy, origin]
        .into_iter()
        .flatten()
        .find(|dir| dir.join("config.json").is_file());
    let modelfile_path = project.join(MODELFILE);
    std::fs::write(&modelfile_path, modelfile(&artifact, config))
        .map_err(|e| format!("{}: {e}", modelfile_path.display()))?;
    let name = model_name(project_name);
    let shown = plain_path(modelfile_path.clone()).display().to_string();
    Ok(Some(Export {
        command: format!("ollama create {name} -f \"{shown}\""),
        artifact,
        modelfile: modelfile_path,
        name,
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A Qwen2-like folder: ChatML template, Qwen's sampling defaults, 32k context.
    fn model_dir(dir: &Path) -> PathBuf {
        let model = dir.join("copies").join("c1");
        std::fs::create_dir_all(&model).unwrap();
        let write = |name: &str, value: Value| {
            std::fs::write(model.join(name), value.to_string()).unwrap();
        };
        write("config.json", json!({"max_position_embeddings": 32768}));
        write(
            "generation_config.json",
            json!({"do_sample": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                   "repetition_penalty": 1.05, "eos_token_id": [151645, 151643]}),
        );
        write(
            "tokenizer_config.json",
            json!({"chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n\
                                     {{ m.content }}<|im_end|>\n{% endfor %}"}),
        );
        model
    }

    #[test]
    fn the_modelfile_carries_the_model_sampling_context_and_stops() {
        let dir = tempfile::tempdir().unwrap();
        let model = model_dir(dir.path());
        let gguf = Artifact::Gguf(dir.path().join("out").join("model.gguf"));
        let text = modelfile(&gguf, Some(&model));
        let lines: Vec<&str> = text.lines().collect();
        // Quoted when the temporary folder holds a space.
        let from = lines[0].trim_end_matches('"');
        assert!(
            from.starts_with("FROM ") && from.ends_with("out/model.gguf"),
            "{from}"
        );
        assert_eq!(
            lines[1..],
            [
                "PARAMETER temperature 0.7",
                "PARAMETER top_p 0.8",
                "PARAMETER top_k 20",
                "PARAMETER repeat_penalty 1.05",
                "PARAMETER num_ctx 8192",
                "PARAMETER stop \"<|im_end|>\"",
            ]
        );
        // A GGUF model alone (no folder to read): the artifact and nothing else.
        assert_eq!(modelfile(&gguf, None).lines().count(), 1);
    }

    #[test]
    fn a_path_with_a_space_is_quoted() {
        let text = modelfile(&Artifact::Gguf(PathBuf::from("my models/m.gguf")), None);
        assert_eq!(text, "FROM \"my models/m.gguf\"\n");
    }

    #[test]
    fn the_newest_gguf_wins_else_the_working_copy_folder() {
        let dir = tempfile::tempdir().unwrap();
        let copy = model_dir(dir.path());
        assert_eq!(
            find_artifact(dir.path(), Some(&copy)),
            Some(Artifact::SafeTensors(copy.clone()))
        );
        assert_eq!(
            find_artifact(dir.path(), Some(&dir.path().join("none"))),
            None
        );
        let out = dir.path().join("runs").join("r1").join("out");
        std::fs::create_dir_all(&out).unwrap();
        let old = out.join("old.gguf");
        std::fs::write(&old, b"GGUF").unwrap();
        let new = dir.path().join("work").join("model.GGUF");
        std::fs::create_dir_all(new.parent().unwrap()).unwrap();
        std::fs::write(&new, b"GGUF").unwrap();
        let past = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(past)
            .unwrap();
        assert_eq!(
            find_artifact(dir.path(), Some(&copy)),
            Some(Artifact::Gguf(new))
        );
    }

    #[test]
    fn model_names_keep_only_what_ollama_accepts() {
        assert_eq!(model_name("demo-qwen2"), "demo-qwen2");
        assert_eq!(model_name("Mon Projet (v2)!"), "mon-projet-v2");
        assert_eq!(model_name("Qwen2.5_7B"), "qwen2.5_7b");
        assert_eq!(model_name("***"), "surgeon-model");
    }

    #[test]
    fn export_writes_the_modelfile_in_the_project_and_names_the_command() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(export(dir.path(), "p", None, None, None), Ok(None));
        let copy = model_dir(dir.path());
        let done = export(dir.path(), "Demo Qwen2", None, Some(&copy), None)
            .unwrap()
            .unwrap();
        assert_eq!(done.artifact, Artifact::SafeTensors(copy));
        assert_eq!(done.name, "demo-qwen2");
        let written = std::fs::read_to_string(dir.path().join(MODELFILE)).unwrap();
        assert!(
            written.contains("PARAMETER stop \"<|im_end|>\""),
            "{written}"
        );
        assert!(done.command.starts_with("ollama create demo-qwen2 -f \""));
        assert!(done.command.ends_with("Modelfile\""), "{}", done.command);
    }

    #[test]
    fn a_fresh_quantized_export_is_the_modelfile_source() {
        let dir = tempfile::tempdir().unwrap();
        let copy = model_dir(dir.path());
        let fresh = dir.path().join("runs").join("r1").join("out");
        std::fs::create_dir_all(&fresh).unwrap();
        let fresh = fresh.join("c1-Q4_K_M.gguf");
        std::fs::write(&fresh, b"GGUF").unwrap();
        let past = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&fresh)
            .unwrap()
            .set_modified(past)
            .unwrap();
        std::fs::write(dir.path().join("newer.gguf"), b"GGUF").unwrap();
        let done = export(
            dir.path(),
            "p",
            Some(Artifact::Gguf(fresh.clone())),
            Some(&copy),
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(done.artifact, Artifact::Gguf(fresh));
        let written = std::fs::read_to_string(dir.path().join(MODELFILE)).unwrap();
        let from = written.lines().next().unwrap().trim_end_matches('"');
        assert!(from.ends_with("runs/r1/out/c1-Q4_K_M.gguf"), "{from}");
        assert!(written.contains("PARAMETER num_ctx 8192"), "{written}");
    }
}
