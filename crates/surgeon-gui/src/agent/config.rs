//! What the Agents page remembers, never a secret: the connections (`agents.json` next to
//! `settings.json`), the consents, the local-only switch, and per project (`chat/agent.json`)
//! the chosen connection and model, the modes, the budgets and the "always" rules.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::anthropic::{self, Anthropic};
use super::http::Reach;
use super::openai::{self, OpenAi};
use super::policy::{AgentMode, Rule};
use super::provider::{LlmProvider, ProviderError};
use super::vault::SecretStore;
use crate::studio::project::write_atomic;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    /// Any OpenAI-compatible endpoint: OpenRouter, Ollama, LM Studio, `llama-server`.
    Compatible,
    /// The Claude Code CLI, on the claude.ai account it is signed in with: its own loop.
    ClaudeCode,
    /// The Codex CLI, signed in by the user.
    Codex,
    /// The OpenCode CLI, with its own tools kept (the source model is held by the kernel lock).
    OpenCode,
    /// The Antigravity CLI (`agy`), in an isolated profile, its own tools kept.
    Antigravity,
}

impl ProviderKind {
    pub const ALL: [Self; 7] = [
        Self::Anthropic,
        Self::OpenAi,
        Self::Compatible,
        Self::ClaudeCode,
        Self::Codex,
        Self::OpenCode,
        Self::Antigravity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic (clé d'API)",
            Self::OpenAi => "OpenAI (clé d'API)",
            Self::Compatible => "Compatible OpenAI (URL)",
            Self::ClaudeCode => "Claude Code (CLI installée)",
            Self::Codex => "Codex CLI (installée)",
            Self::OpenCode => "OpenCode CLI (installée)",
            Self::Antigravity => "Antigravity CLI (installée)",
        }
    }

    /// The product name alone, for sentences ("needed by Claude Code").
    pub fn name(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::OpenAi => "OpenAI",
            Self::Compatible => "OpenAI-compatible",
            Self::ClaudeCode => "Claude Code",
            Self::Codex => "Codex",
            Self::OpenCode => "OpenCode",
            Self::Antigravity => "Antigravity",
        }
    }

    pub fn is_cli(self) -> bool {
        matches!(
            self,
            Self::ClaudeCode | Self::Codex | Self::OpenCode | Self::Antigravity
        )
    }

    pub fn needs_key(self) -> bool {
        matches!(self, Self::Anthropic | Self::OpenAi)
    }

    /// The `--model` names of a CLI that cannot list its own, offered until a list comes
    /// back. API connections list theirs from the provider, never from here.
    pub fn fallback_models(self) -> &'static [&'static str] {
        match self {
            Self::ClaudeCode => &["sonnet", "opus", "haiku", "fable"],
            Self::Codex => &["gpt-5-codex", "o3", "o4-mini"],
            _ => &[],
        }
    }

    /// The model of a connection nothing was picked for. Claude Code gets one by name: left
    /// to itself, the CLI may settle on a model the account's plan does not include.
    pub fn default_model(self) -> Option<&'static str> {
        match self {
            Self::ClaudeCode => Some("sonnet"),
            _ => None,
        }
    }

    /// A model billed on top of the plan (Fable needs usage credits): it runs when picked for
    /// the project, and is never carried over to one by itself.
    pub fn needs_credits(self, model: &str) -> bool {
        self == Self::ClaudeCode && model.to_ascii_lowercase().contains("fable")
    }

    /// A Claude provider, whose thinking depth the user can set ([`Effort`]).
    pub fn takes_effort(self) -> bool {
        matches!(self, Self::Anthropic | Self::ClaudeCode)
    }

    pub fn default_base(self) -> &'static str {
        match self {
            Self::Anthropic => anthropic::DEFAULT_BASE,
            Self::OpenAi => openai::DEFAULT_BASE,
            _ => "",
        }
    }
}

/// How hard a Claude model thinks. `Off` turns thinking off where the model allows it, else
/// asks for the lowest effort; no value at all leaves the model's own default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Off,
    Low,
    Medium,
    High,
    Max,
}

impl Effort {
    pub const ALL: [Self; 5] = [Self::Off, Self::Low, Self::Medium, Self::High, Self::Max];

    /// The API and `claude --effort` name; `Off` has none.
    pub fn level(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
            Self::Max => Some("max"),
        }
    }
}

/// Prices the user typed, in dollars per million tokens. None is hard-coded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Prices {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

impl Prices {
    pub fn cost(&self, usage: super::provider::Usage) -> f64 {
        (usage.input_tokens as f64 * self.input_per_mtok
            + usage.output_tokens as f64 * self.output_per_mtok)
            / 1e6
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Connection {
    pub id: String,
    pub kind: ProviderKind,
    pub label: String,
    /// Empty: the provider's default.
    #[serde(default)]
    pub base_url: String,
    /// The last list the provider's API returned.
    #[serde(default)]
    pub models: Vec<String>,
    /// The model a project gets when it does not pick one.
    #[serde(default)]
    pub default_model: Option<String>,
    /// The model last picked with this connection, picked again when a project switches to it.
    #[serde(default)]
    pub last_model: Option<String>,
    /// The thinking depth of a Claude connection; `None`: the model's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// A CLI runtime whose sign-in check passed (Tester).
    #[serde(default)]
    pub verified: bool,
    #[serde(default)]
    pub prices: BTreeMap<String, Prices>,
}

impl Connection {
    pub fn base(&self) -> &str {
        match self.base_url.trim() {
            "" => self.kind.default_base(),
            b => b,
        }
    }

    /// Offered to the sessions: a CLI whose sign-in check passed, an API connection whose
    /// model list came back.
    pub fn ready(&self) -> bool {
        match self.kind.is_cli() {
            true => self.verified,
            false => !self.models.is_empty(),
        }
    }

    /// The model a project inherits: the chosen default, else the first one listed, else the
    /// default of its kind.
    pub fn inherited_model(&self) -> Option<&str> {
        self.default_model
            .as_deref()
            .or(self.models.first().map(String::as_str))
            .or(self.kind.default_model())
    }

    /// The models the pickers offer: the last list, else the CLI's built-in names.
    pub fn model_choices(&self) -> Vec<String> {
        match self.models.is_empty() {
            true => self
                .kind
                .fallback_models()
                .iter()
                .map(|m| (*m).to_owned())
                .collect(),
            false => self.models.clone(),
        }
    }

    /// The model a switch to this connection lands on: the last one picked with it unless it
    /// costs credits, else the one it hands down (`None`: a CLI's own default, no `--model`).
    pub fn preferred_model(&self) -> Option<&str> {
        self.last_model
            .as_deref()
            .filter(|m| !self.kind.needs_credits(m))
            .or(self.inherited_model())
    }

    /// Whether requests stay on this machine (CLI runtimes reach their own servers).
    pub fn is_local(&self) -> bool {
        !self.kind.is_cli() && super::http::is_local_url(self.base())
    }

    /// The provider of an API connection; its key comes from `store`.
    pub fn provider(
        &self,
        store: &dyn SecretStore,
        local_only: bool,
    ) -> Result<Arc<dyn LlmProvider>, ProviderError> {
        let reach = match local_only {
            true => Reach::LocalOnly,
            false => Reach::Network,
        };
        let key = store.get(&self.id).map_err(ProviderError::Blocked)?;
        Ok(match self.kind {
            ProviderKind::Anthropic => Arc::new(Anthropic::new(
                self.base(),
                key.ok_or_else(|| ProviderError::Auth("aucune clé enregistrée".into()))?,
                reach,
            )?),
            ProviderKind::OpenAi => Arc::new(OpenAi::new(
                self.base(),
                Some(key.ok_or_else(|| ProviderError::Auth("aucune clé enregistrée".into()))?),
                reach,
            )?),
            ProviderKind::Compatible => Arc::new(OpenAi::new(self.base(), key, reach)?),
            ProviderKind::ClaudeCode
            | ProviderKind::Codex
            | ProviderKind::OpenCode
            | ProviderKind::Antigravity => {
                return Err(ProviderError::Blocked(
                    "un runtime CLI n'est pas un fournisseur par API".into(),
                ))
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Consent {
    pub connection: String,
    pub accepted_unix: u64,
    /// Replace the project and home folders by markers in what is sent.
    pub redact_paths: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentsConfig {
    pub schema_version: u32,
    #[serde(default)]
    pub connections: Vec<Connection>,
    #[serde(default)]
    pub consents: Vec<Consent>,
    /// No request leaves the machine: loopback endpoints only, no CLI runtime. The setting
    /// lives in `settings.json` since G6; an older file is read once for it, never written.
    #[serde(default, skip_serializing)]
    pub local_only: bool,
    /// The connection every project uses unless it picks another.
    #[serde(default)]
    pub default_connection: Option<String>,
}

impl Default for AgentsConfig {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            connections: Vec::new(),
            consents: Vec::new(),
            local_only: false,
            default_connection: None,
        }
    }
}

impl AgentsConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let cfg: Self =
                    serde_json::from_str(&text).map_err(|e| format!("{} : {e}", path.display()))?;
                match cfg.schema_version {
                    SCHEMA_VERSION => Ok(cfg),
                    v => Err(format!("{} : schema_version {v} inconnue", path.display())),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{} : {e}", path.display())),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        write_atomic(path, &bytes)
    }

    /// Hands an older file's `local_only` to `setting` (true wins), then follows it. `true`
    /// when the setting changed and must be saved.
    pub fn adopt_local_only(&mut self, setting: &mut bool) -> bool {
        let migrated = self.local_only && !*setting;
        *setting |= self.local_only;
        self.local_only = *setting;
        migrated
    }

    pub fn connection(&self, id: &str) -> Option<&Connection> {
        self.connections.iter().find(|c| c.id == id)
    }

    pub fn consent(&self, id: &str) -> Option<&Consent> {
        self.consents.iter().find(|c| c.connection == id)
    }

    /// A free id from a label: lower case, `-` for the rest, unique.
    pub fn new_id(&self, label: &str) -> String {
        let base: String = label
            .chars()
            .map(|c| match c.is_ascii_alphanumeric() {
                true => c.to_ascii_lowercase(),
                false => '-',
            })
            .collect::<String>()
            .trim_matches('-')
            .chars()
            .take(32)
            .collect();
        let base = if base.is_empty() {
            "connexion".into()
        } else {
            base
        };
        (1..)
            .map(|n| match n {
                1 => base.clone(),
                n => format!("{base}-{n}"),
            })
            .find(|id| self.connection(id).is_none())
            .unwrap_or(base)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AskMode {
    /// `ask_user` waits for the user.
    #[default]
    AskMe,
    /// `ask_user` is answered at once: take the most careful option.
    DecideAlone,
}

/// Limits of one execution (one user message and everything the agent does after it).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RunBudgets {
    pub max_tokens: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub max_secs: Option<u64>,
    pub max_tool_calls: Option<u32>,
}

impl Default for RunBudgets {
    fn default() -> Self {
        Self {
            max_tokens: Some(300_000),
            max_cost_usd: None,
            max_secs: Some(1800),
            max_tool_calls: Some(25),
        }
    }
}

/// `chat/agent.json` of a project.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProjectAgent {
    #[serde(default)]
    pub connection: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub mode: AgentMode,
    #[serde(default)]
    pub ask: AskMode,
    #[serde(default)]
    pub budgets: RunBudgets,
    #[serde(default)]
    pub rules: Vec<Rule>,
}

impl ProjectAgent {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        write_atomic(path, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_only_moves_to_the_settings_once_and_is_no_longer_written_here() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("agents.json");
        std::fs::write(&path, r#"{"schema_version":1,"local_only":true}"#).unwrap();
        let mut cfg = AgentsConfig::load(&path).unwrap();
        let mut setting = false;
        assert!(
            cfg.adopt_local_only(&mut setting),
            "migrated: save the settings"
        );
        assert!(setting && cfg.local_only);
        cfg.save(&path).unwrap();
        assert!(!std::fs::read_to_string(&path)
            .unwrap()
            .contains("local_only"));
        let mut cfg = AgentsConfig::load(&path).unwrap();
        assert!(
            !cfg.adopt_local_only(&mut setting),
            "nothing more to migrate"
        );
        assert!(cfg.local_only, "follows the setting");
        setting = false;
        let mut cfg = AgentsConfig::load(&path).unwrap();
        cfg.adopt_local_only(&mut setting);
        assert!(!cfg.local_only, "the setting decides once migrated");
    }
}
