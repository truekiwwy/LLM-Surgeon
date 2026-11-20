//! External harnesses (Claude Code, Codex, OpenCode, Cursor) that drive the engine with their
//! own agent: the `weight-surgeon` skill (`skills/weight-surgeon/`, embedded in this build)
//! installed where they look for skills, and the MCP server entry that starts `surgeon mcp`.
//! Nothing is written without a click or a command-line flag. A skill file is written through
//! a temporary file and a rename, `SKILL.md` last; a configuration file keeps every other key
//! and server, and what it held goes to `<file>.bak` first.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::studio::project::write_atomic;

pub const SKILL_NAME: &str = "weight-surgeon";
/// The name of the server in an MCP configuration.
pub const SERVER_NAME: &str = "weight-surgeon";

/// The skill as this build ships it: (path in the skill folder, content). `SKILL.md` comes
/// last, so that an install cut short never reads as up to date.
pub const SKILL_FILES: [(&str, &str); 3] = [
    (
        "references/workflow.md",
        include_str!("../../../skills/weight-surgeon/references/workflow.md"),
    ),
    (
        "references/recipes.md",
        include_str!("../../../skills/weight-surgeon/references/recipes.md"),
    ),
    (
        "SKILL.md",
        include_str!("../../../skills/weight-surgeon/SKILL.md"),
    ),
];

/// A harness and the skills folder it reads under the user's home.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Harness {
    /// `~/.claude/skills` (Claude Code).
    Claude,
    /// `~/.codex/skills` (Codex CLI).
    Codex,
    /// `~/.agents/skills`, the shared folder of the Agent Skills format (OpenCode...).
    Agents,
}

impl Harness {
    pub const ALL: [Self; 3] = [Self::Claude, Self::Codex, Self::Agents];

    /// The product name, the same in every language.
    pub fn name(self) -> &'static str {
        match self {
            Self::Claude => "Claude Code",
            Self::Codex => "Codex",
            Self::Agents => "Agent Skills",
        }
    }

    /// `<home>/<.claude|.codex|.agents>/skills/weight-surgeon`.
    pub fn skill_dir(self, home: &Path) -> PathBuf {
        let root = match self {
            Self::Claude => ".claude",
            Self::Codex => ".codex",
            Self::Agents => ".agents",
        };
        home.join(root).join("skills").join(SKILL_NAME)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillState {
    Missing,
    UpToDate,
    /// Installed, but a file differs from this build's (older, or edited by hand).
    Outdated,
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// The state of the skill folder `dir`, by the SHA-256 of each embedded file.
pub fn skill_state(dir: &Path) -> SkillState {
    if !dir.join("SKILL.md").is_file() {
        return SkillState::Missing;
    }
    let same = SKILL_FILES.iter().all(|(rel, text)| {
        std::fs::read(dir.join(rel)).is_ok_and(|b| sha256(&b) == sha256(text.as_bytes()))
    });
    match same {
        true => SkillState::UpToDate,
        false => SkillState::Outdated,
    }
}

/// Writes this build's skill into `dir` (created), file by file. Files of the folder the skill
/// does not ship are left alone.
pub fn install_skill(dir: &Path) -> Result<(), String> {
    for (rel, text) in SKILL_FILES {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{} : {e}", parent.display()))?;
        }
        write_atomic(&path, text.as_bytes())?;
    }
    Ok(())
}

/// `surgeon` next to this app (the packages ship both), else on `path`; `None` when neither.
fn surgeon_near(exe: Option<&Path>, path: Option<&OsStr>) -> Option<PathBuf> {
    let name = format!("surgeon{}", std::env::consts::EXE_SUFFIX);
    exe.and_then(Path::parent)
        .map(|dir| dir.join(&name))
        .filter(|p| p.is_file())
        .or_else(|| crate::agent::runtime::find_on_path("surgeon", path))
}

/// The program an MCP configuration starts: [`surgeon_near`] this app, else this app itself,
/// which serves `mcp` as `surgeon mcp` does (`runner::serve_mcp`).
pub fn surgeon_command() -> PathBuf {
    let exe = std::env::current_exe().ok();
    let path = std::env::var_os("PATH");
    surgeon_near(exe.as_deref(), path.as_deref())
        .or(exe)
        .unwrap_or_else(|| PathBuf::from("surgeon"))
}

/// The server entry: `command` with `mcp`, which serves the harness's working folder.
pub fn mcp_entry(command: &Path) -> Value {
    json!({"command": command, "args": ["mcp"]})
}

/// The configuration to paste: `{"mcpServers": {"weight-surgeon": …}}`.
pub fn mcp_snippet(command: &Path) -> Value {
    json!({ "mcpServers": { SERVER_NAME: mcp_entry(command) } })
}

/// The user-scope configuration of Claude Code, whose `mcpServers` every project sees.
pub fn claude_config(home: &Path) -> PathBuf {
    home.join(".claude.json")
}

/// Adds or replaces the `weight-surgeon` server of the MCP configuration `file` (created when
/// absent), every other key and server kept (keys may come back in another order). A file that
/// is not a JSON object, or whose `mcpServers` is not one, is refused and left untouched. The
/// `.bak` written first, when the file existed.
pub fn merge_mcp_config(file: &Path, command: &Path) -> Result<Option<PathBuf>, String> {
    let shown = |e: &dyn std::fmt::Display| format!("{} : {e}", file.display());
    let existing = match std::fs::read_to_string(file) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(shown(&e)),
    };
    let mut config: Value = match &existing {
        Some(text) => serde_json::from_str(text).map_err(|e| shown(&e))?,
        None => json!({}),
    };
    let servers = config
        .as_object_mut()
        .ok_or_else(|| shown(&"pas un objet JSON"))?
        .entry("mcpServers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| shown(&"mcpServers n'est pas un objet JSON"))?;
    servers.insert(SERVER_NAME.into(), mcp_entry(command));
    let bytes = serde_json::to_vec_pretty(&config).map_err(|e| shown(&e))?;
    let backup = match existing {
        Some(text) => {
            let mut bak = file.as_os_str().to_owned();
            bak.push(".bak");
            let bak = PathBuf::from(bak);
            std::fs::write(&bak, text).map_err(|e| format!("{} : {e}", bak.display()))?;
            Some(bak)
        }
        None => None,
    };
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(|e| shown(&e))?;
    }
    write_atomic(file, &bytes)?;
    Ok(backup)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A home of its own: nothing here reads or writes the user's.
    fn home() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn the_skill_goes_missing_installed_outdated_and_up_to_date_again() {
        let home = home();
        for harness in Harness::ALL {
            let dir = harness.skill_dir(home.path());
            assert!(dir.starts_with(home.path()));
            assert_eq!(skill_state(&dir), SkillState::Missing, "{harness:?}");
            install_skill(&dir).unwrap();
            assert_eq!(skill_state(&dir), SkillState::UpToDate);
            for (rel, text) in SKILL_FILES {
                assert_eq!(std::fs::read_to_string(dir.join(rel)).unwrap(), text);
            }
            // Edited by hand, or left by an older build: a reference is enough.
            std::fs::write(dir.join("references/recipes.md"), "old").unwrap();
            assert_eq!(skill_state(&dir), SkillState::Outdated);
            std::fs::remove_file(dir.join("references/workflow.md")).unwrap();
            assert_eq!(skill_state(&dir), SkillState::Outdated, "a file missing");
            std::fs::write(dir.join("notes.md"), "mine").unwrap();
            install_skill(&dir).unwrap();
            assert_eq!(skill_state(&dir), SkillState::UpToDate);
            assert_eq!(
                std::fs::read_to_string(dir.join("notes.md")).unwrap(),
                "mine"
            );
            let leftovers = std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
                .count();
            assert_eq!(leftovers, 0, "no temporary file left");
        }
        assert_eq!(
            Harness::Claude.skill_dir(Path::new("h")),
            Path::new("h/.claude/skills/weight-surgeon")
        );
    }

    #[test]
    fn the_embedded_skill_is_the_one_of_the_repository() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../skills/weight-surgeon");
        for (rel, text) in SKILL_FILES {
            assert_eq!(std::fs::read_to_string(root.join(rel)).unwrap(), text);
        }
        assert!(SKILL_FILES
            .last()
            .unwrap()
            .1
            .contains("name: weight-surgeon"));
    }

    #[test]
    fn the_snippet_starts_surgeon_mcp_by_its_absolute_path() {
        let command = Path::new(if cfg!(windows) {
            r"C:\Program Files\Weight Surgeon\surgeon.exe"
        } else {
            "/opt/weight-surgeon/surgeon"
        });
        let snippet = mcp_snippet(command);
        assert_eq!(
            snippet,
            json!({"mcpServers": {"weight-surgeon": {
                "command": command.to_str().unwrap(), "args": ["mcp"]}}})
        );
        // What `surgeon mcp` parses: the args as the harness passes them.
        let argv = std::iter::once("surgeon").chain(["mcp"]);
        let cli = <weight_surgeon::cli::Cli as clap::Parser>::try_parse_from(argv).unwrap();
        assert!(matches!(cli.command, weight_surgeon::cli::Commands::Mcp(_)));
    }

    #[test]
    fn surgeon_is_found_next_to_the_app_then_on_the_path() {
        let dir = home();
        let app = dir.path().join("app");
        let bin = dir.path().join("bin");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::create_dir_all(&bin).unwrap();
        let exe = app.join("surgeon-gui");
        let name = format!("surgeon{}", std::env::consts::EXE_SUFFIX);
        let path = std::env::join_paths([&bin]).unwrap();
        assert_eq!(surgeon_near(Some(&exe), Some(&path)), None);
        std::fs::write(bin.join(&name), b"").unwrap();
        assert_eq!(surgeon_near(Some(&exe), Some(&path)), Some(bin.join(&name)));
        std::fs::write(app.join(&name), b"").unwrap();
        assert_eq!(surgeon_near(Some(&exe), Some(&path)), Some(app.join(&name)));
    }

    #[test]
    fn merging_keeps_every_other_server_and_backs_the_file_up() {
        let home = home();
        let file = claude_config(home.path());
        let command = Path::new("/opt/surgeon");
        // Absent: created, nothing to back up.
        assert_eq!(merge_mcp_config(&file, command).unwrap(), None);
        let created: Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(created, mcp_snippet(command));

        let before = json!({
            "numStartups": 42,
            "projects": {"/work": {"allowedTools": ["Bash"]}},
            "mcpServers": {
                "github": {"command": "gh-mcp", "args": ["serve"], "env": {"X": "1"}},
                "weight-surgeon": {"command": "/old/surgeon", "args": ["mcp", "--read-only"]}
            }
        });
        let text = serde_json::to_string_pretty(&before).unwrap();
        std::fs::write(&file, &text).unwrap();
        let bak = merge_mcp_config(&file, command).unwrap().unwrap();
        assert_eq!(
            std::fs::read_to_string(&bak).unwrap(),
            text,
            "the old file, as it was"
        );
        let after: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let mut expected = before.clone();
        expected["mcpServers"]["weight-surgeon"] = mcp_entry(command);
        assert_eq!(after, expected, "only our server changed");

        // Twice: the same file.
        merge_mcp_config(&file, command).unwrap();
        let again: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(again, expected);

        // Not a configuration it understands: refused, untouched.
        for broken in ["[1, 2]", "{\"mcpServers\": []}", "{oops"] {
            std::fs::write(&file, broken).unwrap();
            assert!(merge_mcp_config(&file, command).is_err(), "{broken}");
            assert_eq!(std::fs::read_to_string(&file).unwrap(), broken);
        }
    }
}
