//! Choosing the mistral.rs archive from the GitHub releases API. Pure: the JSON comes in, the
//! plan comes out; the network lives in [`crate::net`].
//!
//! The releases carry Python wheels and binary archives; the app installs the CPU zip of its
//! own target (`mistralrs-cpu-<triple>.zip`, a single `mistralrs` binary). The list is scanned
//! newest first for the first release that has it.

use serde::Deserialize;

/// The build the app installs: the official CPU archive (GPU builds come from a CUDA build or
/// by hand, see `backend_choice`).
pub const VARIANT: &str = "cpu";

/// The target triple the release names its archives with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub triple: &'static str,
}

impl Target {
    pub fn current() -> Self {
        Self {
            triple: match (std::env::consts::OS, std::env::consts::ARCH) {
                ("windows", _) => "x86_64-pc-windows-msvc",
                ("macos", _) => "aarch64-apple-darwin",
                (_, "aarch64") => "aarch64-unknown-linux-gnu",
                _ => "x86_64-unknown-linux-gnu",
            },
        }
    }

    /// Only the Windows zip archive is installed by the app; elsewhere the wizard gives
    /// manual instructions (`tar.gz` archives).
    pub fn supported(self) -> bool {
        self.triple.ends_with("windows-msvc")
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Asset {
    pub name: String,
    pub size: u64,
    #[serde(default)]
    pub digest: Option<String>,
    pub browser_download_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub draft: bool,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

/// What would be downloaded, shown to the user before anything is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub tag: String,
    pub name: String,
    pub url: String,
    pub size: u64,
    /// The SHA-256 the release publishes (`digest: "sha256:..."`), when it does.
    pub sha256: Option<String>,
    pub variant: &'static str,
}

/// `sha256:<64 hex>` as the API writes it; anything else is no digest.
pub fn parse_digest(digest: &str) -> Option<String> {
    let hex = digest.strip_prefix("sha256:")?;
    (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| hex.to_ascii_lowercase())
}

/// The newest release that carries the CPU zip of `target`.
pub fn select(releases: &[Release], target: Target) -> Option<Plan> {
    let wanted = format!("mistralrs-{VARIANT}-{}.zip", target.triple);
    releases.iter().filter(|r| !r.draft).find_map(|r| {
        let asset = r.assets.iter().find(|a| a.name == wanted)?;
        Some(Plan {
            tag: r.tag_name.clone(),
            name: asset.name.clone(),
            url: asset.browser_download_url.clone(),
            size: asset.size,
            sha256: asset.digest.as_deref().and_then(parse_digest),
            variant: VARIANT,
        })
    })
}

pub fn parse_releases(json: &[u8]) -> Result<Vec<Release>, String> {
    serde_json::from_slice(json)
        .map_err(|e| format!("réponse de l'API des releases illisible : {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIN: Target = Target {
        triple: "x86_64-pc-windows-msvc",
    };

    fn release(tag: &str, assets: &[(&str, Option<String>)]) -> Release {
        Release {
            tag_name: tag.into(),
            draft: false,
            assets: assets
                .iter()
                .map(|(n, d)| Asset {
                    name: (*n).into(),
                    size: 10,
                    digest: d.clone(),
                    browser_download_url: format!("https://github.com/x/{n}"),
                })
                .collect(),
        }
    }

    #[test]
    fn picks_the_cpu_zip_of_the_target_and_skips_look_alikes() {
        let digest = format!("sha256:{}", "AB".repeat(32));
        let rs = [release(
            "v0.9.5",
            &[
                (
                    "mistralrs-0.9.5+cuda128.sm80-cp310-abi3-manylinux_2_35_x86_64.whl",
                    None,
                ),
                ("mistralrs-cpu-x86_64-unknown-linux-gnu.tar.gz", None),
                ("mistralrs-metal-aarch64-apple-darwin.tar.gz", None),
                ("mistralrs-cpu-x86_64-pc-windows-msvc.zip", Some(digest)),
            ],
        )];
        let plan = select(&rs, WIN).unwrap();
        assert_eq!(plan.name, "mistralrs-cpu-x86_64-pc-windows-msvc.zip");
        assert_eq!(
            (plan.sha256, plan.tag.as_str()),
            (Some("ab".repeat(32)), "v0.9.5")
        );
        let mac = Target {
            triple: "aarch64-apple-darwin",
        };
        assert!(
            select(&rs, mac).is_none(),
            "no zip there: installed by hand"
        );
        assert!(WIN.supported() && !mac.supported());
    }

    #[test]
    fn skips_releases_without_the_archive_and_drafts() {
        let mut draft = release(
            "v0.9.6",
            &[("mistralrs-cpu-x86_64-pc-windows-msvc.zip", None)],
        );
        draft.draft = true;
        let rs = [
            draft,
            release(
                "v0.9.5-py",
                &[("mistralrs-0.9.5-cp310-abi3-win_amd64.whl", None)],
            ),
            release(
                "v0.9.4",
                &[("mistralrs-cpu-x86_64-pc-windows-msvc.zip", None)],
            ),
        ];
        assert_eq!(select(&rs, WIN).unwrap().tag, "v0.9.4");
    }

    #[test]
    fn digests_are_validated() {
        assert!(parse_digest("sha256:abc").is_none());
        assert!(parse_digest(&format!("md5:{}", "a".repeat(32))).is_none());
        assert!(parse_digest(&format!("sha256:{}", "g".repeat(64))).is_none());
        assert!(parse_digest(&format!("sha256:{}", "a".repeat(64))).is_some());
    }
}
