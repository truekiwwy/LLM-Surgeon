//! The approval policy. It is computed by this code from the tool name and the parsed
//! arguments only: the agent's prose, the tool results and the model outputs have no say.
//! `Safe` runs, `Approval` waits for a human, `Forbidden` is answered with an error.
//!
//! - Forbidden: an unknown tool, a path outside the sandbox (project folder and source
//!   model folder), anything the job manager refuses (the source model written, an in-place
//!   tool on anything but a working copy), a pipeline without its Safety pillar, a promotion
//!   (human only), and in "Ask" mode every tool that is not a read.
//! - Approval: in "Modify" mode, every write, external program (llama.cpp) or job not
//!   proven read-only. Destructive ones (in-place apply, LoRA merge, pipeline cancel) can
//!   never be allowed "always".

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::tools::UI_TOOLS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// Read only: check, inspect, graph, bench on working copies.
    #[default]
    Ask,
    /// May propose and run writes on working copies, each one approved.
    Modify,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Safe,
    Approval {
        reasons: Vec<String>,
        /// False for destructive actions: "always for this project" is not offered.
        allow_always: bool,
    },
    Forbidden(String),
}

/// Where the agent may act.
pub struct Scope<'a> {
    pub project_dir: &'a Path,
    pub copies_dir: PathBuf,
    /// The source model (file or folder).
    pub source: Option<PathBuf>,
    /// The job manager's own guard (`Manager::check`): origin never written, in-place tools
    /// on copies only, pipeline budgets.
    pub guard: &'a dyn Fn(&str, &Value) -> Result<(), String>,
}

impl Scope<'_> {
    fn roots(&self) -> Vec<PathBuf> {
        let mut roots = vec![canonical(self.project_dir)];
        if let Some(s) = &self.source {
            roots.push(canonical(match s.is_dir() {
                true => s,
                false => s.parent().unwrap_or(s),
            }));
        }
        roots
    }

    /// An argument path as the engine will see it (`{out}` lands under `runs/`).
    pub fn resolve(&self, s: &str) -> PathBuf {
        let s = s.replace(crate::studio::jobs::OUT, "runs/_out");
        let p = Path::new(&s);
        canonical(&match p.is_absolute() {
            true => p.to_path_buf(),
            false => self.project_dir.join(p),
        })
    }
}

pub fn classify(tool: &str, args: &Value, mode: AgentMode, scope: &Scope) -> Decision {
    if UI_TOOLS.contains(&tool) {
        return Decision::Safe;
    }
    let known = weight_surgeon::mcp::tool_definitions()
        .as_array()
        .is_some_and(|t| t.iter().any(|d| d["name"] == tool));
    if !known {
        return Decision::Forbidden(format!("outil inconnu : {tool}"));
    }
    if !args.is_object() {
        return Decision::Forbidden("arguments illisibles (objet JSON attendu)".into());
    }
    let roots = scope.roots();
    for s in strings(args) {
        if looks_like_path(s) {
            let p = scope.resolve(s);
            if !roots.iter().any(|r| p.starts_with(r)) {
                return Decision::Forbidden(format!("hors sandbox : {s}"));
            }
        }
    }
    if let Err(e) = (scope.guard)(tool, args) {
        return Decision::Forbidden(e);
    }
    let action = args["action"].as_str().unwrap_or("");
    if tool == "surgeon_pipeline" {
        if action == "promote_patch" {
            return Decision::Forbidden(
                "promotion d'un résultat : action humaine seulement (onglet Projets)".into(),
            );
        }
        if let Some(config) = args["config"].as_str() {
            let text = std::fs::read_to_string(scope.resolve(config)).unwrap_or_default();
            if let Ok(cfg) = serde_json::from_str::<Value>(&text) {
                let safety = cfg["pillars"]
                    .as_array()
                    .is_some_and(|p| p.iter().any(|p| p["kind"] == "safety"));
                if !safety {
                    return Decision::Forbidden(
                        "pipeline sans pilier safety : désactiver la porte Sécurité est interdit"
                            .into(),
                    );
                }
            }
        }
    }
    let effect = effect(tool, args);
    match (mode, &effect) {
        (AgentMode::Modify, Effect::Read) => Decision::Safe,
        (AgentMode::Ask, Effect::Read) => {
            let on_copy = |key: &str| {
                args[key]
                    .as_str()
                    .is_none_or(|m| scope.resolve(m).starts_with(canonical(&scope.copies_dir)))
            };
            match tool != "surgeon_bench" || on_copy("model") {
                true => Decision::Safe,
                false => Decision::Forbidden(
                    "mode Demander : surgeon_bench seulement sur une copie de travail (copies/)"
                        .into(),
                ),
            }
        }
        (AgentMode::Ask, _) => Decision::Forbidden(format!(
            "mode Demander (lecture seule) : {tool} {}",
            effect.reasons().join(", ")
        )),
        (_, e) => Decision::Approval {
            reasons: e.reasons(),
            allow_always: !matches!(e, Effect::Destructive(_)),
        },
    }
}

enum Effect {
    Read,
    Write(Vec<String>),
    Destructive(Vec<String>),
}

impl Effect {
    fn reasons(&self) -> Vec<String> {
        match self {
            Self::Read => Vec::new(),
            Self::Write(r) | Self::Destructive(r) => r.clone(),
        }
    }
}

/// What a call does, from the tool and its arguments. Anything not proven a read is a write.
fn effect(tool: &str, args: &Value) -> Effect {
    let has = |k: &str| !args[k].is_null();
    let mode = args["mode"].as_str().unwrap_or("");
    let external = "exécute un programme externe de la chaîne d'outils (llama.cpp)".to_owned();
    let writes = "écrit dans le projet".to_owned();
    match tool {
        "surgeon_check" | "surgeon_inspect" | "surgeon_backends" => Effect::Read,
        "surgeon_graph" if !has("out_dir") && !has("cache_dir") => Effect::Read,
        "surgeon_bench"
            if args
                .as_object()
                .is_some_and(|o| o.keys().any(|k| k.starts_with("judge"))) =>
        {
            Effect::Write(vec![external])
        }
        "surgeon_bench" => match mode {
            "logits" | "refusal" => Effect::Read,
            "eval" if args["backend"] == "native" => Effect::Read,
            "dump" => Effect::Write(vec![writes]),
            _ => Effect::Write(vec![external]),
        },
        "surgeon_apply" if args["dry_run"] == true => Effect::Read,
        "surgeon_apply" => Effect::Destructive(vec![
            "réécrit les poids de la copie de travail (sauvegarde et rollback imposés)".into(),
        ]),
        "surgeon_lora" if mode == "merge" => {
            Effect::Destructive(vec!["fusionne l'adaptateur dans la copie de travail".into()])
        }
        "surgeon_autotune" if !has("out") => Effect::Read,
        "surgeon_eval" => {
            let mut r = vec![external];
            if has("export_gguf") {
                r.push("écrit un GGUF".into());
            }
            Effect::Write(r)
        }
        "surgeon_pipeline" => match args["action"].as_str().unwrap_or("") {
            "status" | "report" => Effect::Read,
            "cancel" => Effect::Destructive(vec!["arrête le job et supprime sa copie".into()]),
            _ => Effect::Write(vec![writes, external]),
        },
        _ => Effect::Write(vec![writes]),
    }
}

/// Every string in a JSON value.
fn strings(v: &Value) -> Vec<&str> {
    match v {
        Value::String(s) => vec![s.as_str()],
        Value::Array(a) => a.iter().flat_map(strings).collect(),
        Value::Object(o) => o.values().flat_map(strings).collect(),
        _ => Vec::new(),
    }
}

/// A relative name without a separator stays in the project folder; anything with one (or
/// a drive, or `..`) is checked.
fn looks_like_path(s: &str) -> bool {
    s.contains(['/', '\\']) || s.starts_with("..") || s.get(1..2) == Some(":")
}

/// Lexically normalized (no `.`, `..` resolved), then canonical for the longest existing
/// ancestor, so that a link or `..` cannot step out of a root.
pub fn canonical(path: &Path) -> PathBuf {
    let mut clean = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                clean.pop();
            }
            Component::CurDir => {}
            other => clean.push(other),
        }
    }
    let mut existing = clean.clone();
    let mut rest = Vec::new();
    loop {
        if let Ok(c) = std::fs::canonicalize(&existing) {
            return rest.iter().rev().fold(c, |acc: PathBuf, n| acc.join(n));
        }
        match (
            existing.file_name().map(|n| n.to_owned()),
            existing.parent(),
        ) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent.to_path_buf();
            }
            _ => return clean,
        }
    }
}

/// "Always for this project": one tool and a narrow pattern of its arguments, dated and
/// revocable. A path inside the project matches by its folder (`copies/c1/*`), every other
/// value exactly; the set of argument names must be the same.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub tool: String,
    pub pattern: BTreeMap<String, String>,
    pub created_unix: u64,
}

impl Rule {
    pub fn from_call(tool: &str, args: &Value, scope: &Scope) -> Self {
        let pattern = args
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| (k.clone(), pattern_of(v, scope)))
            .collect();
        Self {
            tool: tool.into(),
            pattern,
            created_unix: crate::studio::project::now_unix(),
        }
    }

    pub fn matches(&self, tool: &str, args: &Value, scope: &Scope) -> bool {
        let Some(obj) = args.as_object() else {
            return false;
        };
        self.tool == tool
            && obj.len() == self.pattern.len()
            && obj.iter().all(|(k, v)| {
                self.pattern
                    .get(k)
                    .is_some_and(|p| match p.strip_suffix("/*") {
                        Some(dir) => v
                            .as_str()
                            .and_then(|s| relative(scope, s))
                            .is_some_and(|rel| rel.rsplit_once('/').map(|(d, _)| d) == Some(dir)),
                        None => *p == exact(v),
                    })
            })
    }

    pub fn describe(&self) -> String {
        let args: Vec<String> = self
            .pattern
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{} ({})", self.tool, args.join(", "))
    }
}

fn exact(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn pattern_of(v: &Value, scope: &Scope) -> String {
    match v
        .as_str()
        .filter(|s| looks_like_path(s))
        .and_then(|s| relative(scope, s))
    {
        Some(rel) => match rel.rsplit_once('/') {
            Some((dir, _)) => format!("{dir}/*"),
            None => exact(v),
        },
        None => exact(v),
    }
}

/// The path relative to the project folder, with `/`, if it is inside.
fn relative(scope: &Scope, s: &str) -> Option<String> {
    let p = scope.resolve(s);
    let rel = p.strip_prefix(canonical(scope.project_dir)).ok()?;
    Some(
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
    )
}
