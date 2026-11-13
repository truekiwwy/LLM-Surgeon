//! Where the desktop app keeps its files. One root, overridable for tests and portable use:
//! `SURGEON_GUI_DATA_DIR`, else `%LOCALAPPDATA%\weight-surgeon` (Windows) or
//! `$XDG_DATA_HOME/weight-surgeon` (`~/.local/share` when unset).

use std::path::PathBuf;

pub const DATA_DIR_ENV: &str = "SURGEON_GUI_DATA_DIR";

fn env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// The user's home folder, `None` when the environment does not say.
pub fn home_dir() -> Option<PathBuf> {
    env("USERPROFILE").or_else(|| env("HOME"))
}

pub fn data_dir() -> PathBuf {
    if let Some(dir) = env(DATA_DIR_ENV) {
        return dir;
    }
    let base = match cfg!(windows) {
        true => env("LOCALAPPDATA"),
        false => {
            env("XDG_DATA_HOME").or_else(|| home_dir().map(|h| h.join(".local").join("share")))
        }
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join("weight-surgeon")
}

pub fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

pub fn logs_dir() -> PathBuf {
    data_dir().join("logs")
}

/// Where installed engine components live (`llama.cpp/<tag>-<variant>/`).
pub fn components_dir() -> PathBuf {
    data_dir().join("components")
}

/// The single-instance handshake file.
pub fn instance_path() -> PathBuf {
    data_dir().join("instance.json")
}

/// Suggested workspace: a visible folder in the user's documents, never the install folder.
pub fn default_workspace() -> PathBuf {
    let base = home_dir()
        .map(|h| h.join("Documents"))
        .unwrap_or_else(data_dir);
    base.join("weight-surgeon")
}
