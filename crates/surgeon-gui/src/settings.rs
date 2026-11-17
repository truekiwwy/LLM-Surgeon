//! The desktop settings (`settings.json` under the data folder): language, the first-run
//! wizard's outcome, the workspace folder and the pinned download hashes. Separate from the
//! engine's backend `config.toml`, which the wizard only writes through `backend::config`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    #[default]
    Fr,
    En,
}

impl Lang {
    /// `fr`, `fr_FR.UTF-8`, `fr-FR`... anything else is English.
    pub fn from_locale(locale: &str) -> Self {
        match locale.trim().to_ascii_lowercase().starts_with("fr") {
            true => Self::Fr,
            false => Self::En,
        }
    }

    /// The system language: `LC_ALL` / `LANG`, on Windows the user locale from the registry.
    pub fn system() -> Self {
        let from_env = ["LC_ALL", "LANG"]
            .iter()
            .find_map(|n| std::env::var(n).ok().filter(|v| !v.is_empty()));
        if let Some(locale) = from_env {
            return Self::from_locale(&locale);
        }
        match crate::platform::user_locale() {
            Some(locale) => Self::from_locale(&locale),
            None => Self::Fr,
        }
    }
}

/// The two kinds of agent session: `Scalping`, the user names a precise problem and the agent
/// goes straight to it; `Advance`, the agent scans the whole model and finds the problems.
/// Kept as the kind the new-session form offers first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Scalping,
    Advance,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// `None` until the wizard (or the language toggle) chose one.
    pub language: Option<Lang>,
    pub wizard_done: bool,
    pub workspace: Option<PathBuf>,
    /// Archive name -> SHA-256, for assets whose release gives no digest: pinned at first
    /// use, a later download of the same name must match.
    pub pins: BTreeMap<String, String>,
    /// The user allowed the (manual, never silent) check for a newer version.
    pub check_updates: bool,
    /// Widths of the left and right columns, in points (put back on the grid when read).
    pub left_width: Option<f32>,
    pub right_width: Option<f32>,
    /// The left column folded to its rail (`Ctrl+B`).
    pub left_collapsed: bool,
    pub mode: Mode,
    /// No request leaves the machine: no remote AI, no CLI runtime, no download.
    pub local_only: bool,
    /// The inference backend follows the hardware (`backend_choice`) until one is saved by hand.
    pub backend_auto: bool,
}

impl Settings {
    pub fn lang(&self) -> Lang {
        self.language.unwrap_or_default()
    }

    /// A missing file is the default; an unreadable or corrupt one is an error to show, never
    /// a reason to overwrite it silently.
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read(path) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes).map_err(|e| format!("{} : {e}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{} : {e}", path.display())),
        }
    }

    /// Atomic: written next to the target, then renamed over it.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let fail = |e: std::io::Error| format!("{} : {e}", path.display());
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(fail)?;
        }
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, json).map_err(fail)?;
        std::fs::rename(&tmp, path).map_err(fail)
    }

    /// `--reset-settings`: back to the first-run state. Components and the engine's
    /// `config.toml` are left alone.
    pub fn reset(path: &Path) -> Result<(), String> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("{} : {e}", path.display())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_reset_and_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("settings.json");
        assert_eq!(Settings::load(&path).unwrap(), Settings::default());
        let mut s = Settings {
            language: Some(Lang::En),
            wizard_done: true,
            left_collapsed: true,
            mode: Mode::Advance,
            ..Default::default()
        };
        s.pins.insert("a.zip".into(), "ab".repeat(32));
        s.save(&path).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), s);
        std::fs::write(&path, b"{not json").unwrap();
        assert!(Settings::load(&path).is_err());
        Settings::reset(&path).unwrap();
        Settings::reset(&path).unwrap();
        assert!(!Settings::load(&path).unwrap().wizard_done);
    }

    #[test]
    fn a_file_written_before_g6_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            br#"{"language":"fr","wizard_done":true,"left_width":280.0}"#,
        )
        .unwrap();
        let s = Settings::load(&path).unwrap();
        assert_eq!((s.mode, s.left_collapsed), (Mode::Scalping, false));
    }

    #[test]
    fn locale_maps_to_a_language() {
        assert_eq!(Lang::from_locale("fr_FR.UTF-8"), Lang::Fr);
        assert_eq!(Lang::from_locale("fr-CA"), Lang::Fr);
        assert_eq!(Lang::from_locale("en-US"), Lang::En);
    }
}
