//! New-version notification. Never silent: the check runs when the user asks (or allowed it
//! in the settings), a newer release is only announced, and the file is downloaded and
//! hash-verified on an explicit click; installing it stays the user's own action.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde::Deserialize;

use crate::net::{Fetcher, NetError, MAX_API_BYTES};
use crate::releases::parse_digest;

/// Releases of this project.
pub const LATEST_URL: &str = "https://api.github.com/repos/truekiwwy/LLM-Surgeon/releases/latest";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Update {
    pub version: String,
    pub name: String,
    pub url: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Deserialize)]
struct RawAsset {
    name: String,
    size: u64,
    #[serde(default)]
    digest: Option<String>,
    browser_download_url: String,
}

#[derive(Deserialize)]
struct RawRelease {
    tag_name: String,
    #[serde(default)]
    assets: Vec<RawAsset>,
}

/// `v1.2.3` -> [1, 2, 3]; a part that is not a number ends the version (`1.2.3-rc1`).
fn version_parts(version: &str) -> Vec<u64> {
    version
        .trim_start_matches('v')
        .split('.')
        .map_while(|p| p.split(['-', '+']).next()?.parse().ok())
        .collect()
}

pub fn is_newer(candidate: &str, current: &str) -> bool {
    let (c, k) = (version_parts(candidate), version_parts(current));
    !c.is_empty() && c > k
}

/// The installer (preferred) or the portable zip of a release, for this platform.
pub fn pick_update(json: &[u8], current: &str) -> Result<Option<Update>, String> {
    let release: RawRelease =
        serde_json::from_slice(json).map_err(|e| format!("réponse illisible : {e}"))?;
    if !is_newer(&release.tag_name, current) {
        return Ok(None);
    }
    let windows = |a: &&RawAsset| a.name.to_ascii_lowercase().contains("windows");
    let asset = release
        .assets
        .iter()
        .filter(windows)
        .find(|a| a.name.ends_with("-setup.exe"))
        .or_else(|| {
            release
                .assets
                .iter()
                .filter(windows)
                .find(|a| a.name.ends_with(".zip"))
        });
    Ok(asset.map(|a| Update {
        version: release.tag_name.trim_start_matches('v').to_owned(),
        name: a.name.clone(),
        url: a.browser_download_url.clone(),
        size: a.size,
        sha256: a.digest.as_deref().and_then(parse_digest),
    }))
}

/// Asks the releases API whether a newer version exists. Nothing is downloaded.
pub fn check(fetcher: &Fetcher, url: &str, current: &str) -> Result<Option<Update>, String> {
    let json = fetcher
        .get_bytes(url, MAX_API_BYTES)
        .map_err(|e: NetError| e.to_string())?;
    pick_update(&json, current)
}

/// Downloads `update` into `dir` after the user's confirmation and checks its hash; a
/// mismatch deletes the file. The caller opens the folder, it never runs the installer.
pub fn download(
    fetcher: &Fetcher,
    update: &Update,
    dir: &Path,
    cancel: &AtomicBool,
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    let target = dir.join(&update.name);
    let part = dir.join(format!("{}.part", update.name));
    let sha = fetcher
        .download(&update.url, &part, Some(update.size), cancel, progress)
        .map_err(|e| e.to_string())?;
    match &update.sha256 {
        Some(expected) if *expected != sha => {
            let _ = std::fs::remove_file(&part);
            Err(format!(
                "SHA-256 incorrect, fichier supprimé (attendu {expected}, reçu {sha})"
            ))
        }
        _ => std::fs::rename(&part, &target)
            .map(|()| target.clone())
            .map_err(|e| format!("{} : {e}", target.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_numerically() {
        assert!(is_newer("v0.10.0", "0.9.5"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.1.0-rc1", "0.1.0"));
        assert!(!is_newer("nightly", "0.1.0"));
    }

    #[test]
    fn the_installer_is_preferred_and_old_releases_are_ignored() {
        let digest = format!("sha256:{}", "ab".repeat(32));
        let json = format!(
            r#"{{"tag_name":"v0.2.0","assets":[
            {{"name":"weight-surgeon-0.2.0-windows-x64.zip","size":5,"browser_download_url":"https://github.com/z"}},
            {{"name":"weight-surgeon-0.2.0-windows-x64-setup.exe","size":9,"digest":"{digest}","browser_download_url":"https://github.com/s"}},
            {{"name":"SHA256SUMS","size":1,"browser_download_url":"https://github.com/h"}}]}}"#
        );
        let update = pick_update(json.as_bytes(), "0.1.0").unwrap().unwrap();
        assert_eq!(update.name, "weight-surgeon-0.2.0-windows-x64-setup.exe");
        assert_eq!(update.version, "0.2.0");
        assert_eq!(update.sha256, Some("ab".repeat(32)));
        assert!(pick_update(json.as_bytes(), "0.2.0").unwrap().is_none());
        assert!(pick_update(b"{", "0.1.0").is_err());
    }
}
