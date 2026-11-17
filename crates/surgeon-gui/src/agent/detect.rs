//! What the machine already offers an agent, found without spending anything: the CLI
//! runtimes on the `PATH`, the OpenAI-compatible servers listening on loopback, and the API
//! keys present in the environment (their presence only: no value is read here).

use std::ffi::OsStr;
use std::time::Duration;

use super::config::ProviderKind;
use super::http::{read_json, Endpoint, Reach, Timeouts};
use super::runtime::{self, Detected};

/// The servers looked for, by their default port on loopback.
pub const SERVERS: [(&str, &str); 3] = [
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llama-server", "http://127.0.0.1:8080/v1"),
];

/// A closed port answers at once; an open one that hangs is given up quickly.
const PROBE: Duration = Duration::from_millis(300);

/// A provider the app can set up from an environment key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Anthropic,
    OpenAi,
    OpenRouter,
}

impl Preset {
    pub const OPENROUTER_BASE: &'static str = "https://openrouter.ai/api/v1";

    pub fn kind(self) -> ProviderKind {
        match self {
            Self::Anthropic => ProviderKind::Anthropic,
            Self::OpenAi => ProviderKind::OpenAi,
            Self::OpenRouter => ProviderKind::Compatible,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::OpenAi => "OpenAI",
            Self::OpenRouter => "OpenRouter",
        }
    }

    /// Empty: the kind's default.
    pub fn base(self) -> &'static str {
        match self {
            Self::OpenRouter => Self::OPENROUTER_BASE,
            _ => "",
        }
    }
}

/// The environment keys looked for; `None`: known, but only usable as a custom endpoint.
const KEYS: [(&str, Option<Preset>); 5] = [
    ("ANTHROPIC_API_KEY", Some(Preset::Anthropic)),
    ("OPENAI_API_KEY", Some(Preset::OpenAi)),
    ("OPENROUTER_API_KEY", Some(Preset::OpenRouter)),
    ("GEMINI_API_KEY", None),
    ("MISTRAL_API_KEY", None),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvKey {
    pub var: &'static str,
    pub preset: Option<Preset>,
}

/// Why a CLI found on the `PATH` is not offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Not supported in this version.
    NotSupported,
}

/// A CLI found but not offered, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Other {
    pub name: &'static str,
    pub version: String,
    pub reason: Reason,
}

/// The CLIs looked for besides the four runtimes.
const OTHERS: [(&str, Reason); 1] = [("gemini", Reason::NotSupported)];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Server {
    pub name: &'static str,
    pub base: &'static str,
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scan {
    pub runtimes: Vec<Detected>,
    pub others: Vec<Other>,
    pub servers: Vec<Server>,
    pub keys: Vec<EnvKey>,
}

/// The known keys `present` says are set.
pub fn env_keys(present: impl Fn(&str) -> bool) -> Vec<EnvKey> {
    KEYS.iter()
        .filter(|(var, _)| present(var))
        .map(|&(var, preset)| EnvKey { var, preset })
        .collect()
}

/// The four runtimes (with their checked options), then the other CLIs.
pub fn clis(path: Option<&OsStr>) -> (Vec<Detected>, Vec<Other>) {
    let runtimes = [
        ProviderKind::ClaudeCode,
        ProviderKind::Codex,
        ProviderKind::OpenCode,
        ProviderKind::Antigravity,
    ]
    .into_iter()
    .filter_map(|k| runtime::detect(k, path))
    .collect();
    let others = OTHERS
        .iter()
        .filter_map(|&(name, reason)| {
            let bin = runtime::find_on_path(name, path)?;
            Some(Other {
                name,
                version: runtime::version(&bin),
                reason,
            })
        })
        .collect();
    (runtimes, others)
}

/// The models a server at `base` lists, or `None` when nothing answers there in time.
pub fn probe(base: &str) -> Option<Vec<String>> {
    let timeouts = Timeouts {
        connect: PROBE,
        first_byte: PROBE,
        body: PROBE * 3,
    };
    let endpoint = Endpoint::new(base, Reach::LocalOnly, timeouts).ok()?;
    let page = read_json(endpoint.get("/models", &[]).ok()?).ok()?;
    super::openai::model_ids(&page)
}

/// Everything at once, each server probed on its own thread.
pub fn scan() -> Scan {
    let probes: Vec<_> = SERVERS
        .iter()
        .map(|&(name, base)| {
            std::thread::spawn(move || probe(base).map(|models| Server { name, base, models }))
        })
        .collect();
    let path = std::env::var_os("PATH");
    let (runtimes, others) = clis(path.as_deref());
    Scan {
        runtimes,
        others,
        servers: probes
            .into_iter()
            .filter_map(|p| p.join().ok().flatten())
            .collect(),
        keys: env_keys(|var| std::env::var_os(var).is_some_and(|v| !v.is_empty())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_present_keys_are_listed_with_their_preset() {
        let keys = env_keys(|v| v == "OPENROUTER_API_KEY" || v == "GEMINI_API_KEY");
        assert_eq!(
            keys,
            [
                EnvKey {
                    var: "OPENROUTER_API_KEY",
                    preset: Some(Preset::OpenRouter)
                },
                EnvKey {
                    var: "GEMINI_API_KEY",
                    preset: None
                },
            ]
        );
        assert!(env_keys(|_| false).is_empty());
        assert_eq!(Preset::OpenRouter.kind(), ProviderKind::Compatible);
        assert!(Preset::OpenRouter.base().starts_with("https://"));
    }

    #[test]
    fn nothing_listening_is_no_server() {
        // A port just released: nothing answers there.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert_eq!(probe(&format!("http://127.0.0.1:{port}/v1")), None);
        assert_eq!(probe("https://example.com/v1"), None, "loopback only");
    }
}
