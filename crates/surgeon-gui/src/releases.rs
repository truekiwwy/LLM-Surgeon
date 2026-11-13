//! Choosing the llama.cpp archive from the GitHub releases API. Pure: the JSON comes in, the
//! plan comes out; the network lives in [`crate::net`].
//!
//! `releases/latest` cannot be used: on ggml-org/llama.cpp it points at a tag with a single
//! `nightly-tag.txt` asset. The list is scanned newest first for the first release that has
//! the wanted archive.

use serde::Deserialize;

/// Which llama.cpp build the operator wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// GPU through Vulkan (AMD Radeon, Intel, NVIDIA).
    Vulkan,
    /// CPU only.
    Cpu,
    /// AMD HIP (ROCm), optional.
    Hip,
}

impl Variant {
    pub fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
            Self::Hip => "hip",
        }
    }
}

/// OS and CPU architecture, as the release names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub os: &'static str,
    pub arch: &'static str,
}

impl Target {
    pub fn current() -> Self {
        Self {
            os: match std::env::consts::OS {
                "windows" => "win",
                "macos" => "macos",
                _ => "ubuntu",
            },
            arch: match std::env::consts::ARCH {
                "aarch64" => "arm64",
                _ => "x64",
            },
        }
    }

    /// Only the Windows zip archives are installed by the app; elsewhere the wizard gives
    /// manual instructions.
    pub fn supported(self) -> bool {
        self.os == "win"
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

/// `llama-<tag>-bin-<os>-<flavor>-<arch>.zip`, flavor being `vulkan`, `cpu` or `rocm-<ver>`.
/// The `cudart-` runtime bundles and the other flavors never match.
fn matches(name: &str, variant: Variant, target: Target) -> bool {
    let suffix = format!("-{}.zip", target.arch);
    let marker = format!("-bin-{}-", target.os);
    let Some(flavor) = name
        .strip_prefix("llama-")
        .and_then(|n| n.strip_suffix(&suffix))
        .and_then(|n| n.split_once(&marker))
        .map(|(_, flavor)| flavor)
    else {
        return false;
    };
    match variant {
        Variant::Vulkan => flavor == "vulkan",
        Variant::Cpu => flavor == "cpu",
        Variant::Hip => flavor.starts_with("rocm-") || flavor == "hip-radeon",
    }
}

/// The newest release that carries the wanted archive.
pub fn select(releases: &[Release], variant: Variant, target: Target) -> Option<Plan> {
    releases.iter().filter(|r| !r.draft).find_map(|r| {
        let asset = r
            .assets
            .iter()
            .find(|a| matches(&a.name, variant, target))?;
        Some(Plan {
            tag: r.tag_name.clone(),
            name: asset.name.clone(),
            url: asset.browser_download_url.clone(),
            size: asset.size,
            sha256: asset.digest.as_deref().and_then(parse_digest),
            variant: variant.label(),
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
        os: "win",
        arch: "x64",
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
    fn picks_the_flavor_and_skips_look_alikes() {
        let digest = format!("sha256:{}", "AB".repeat(32));
        let rs = [release(
            "b1",
            &[
                ("cudart-llama-bin-win-cuda-12.4-x64.zip", None),
                ("llama-b1-bin-win-cuda-12.4-x64.zip", None),
                ("llama-b1-bin-win-cpu-x64.zip", None),
                ("llama-b1-bin-win-vulkan-arm64.zip", None),
                ("llama-b1-bin-win-vulkan-x64.zip", Some(digest)),
                ("llama-b1-bin-win-rocm-10.0-x64.zip", None),
            ],
        )];
        let vk = select(&rs, Variant::Vulkan, WIN).unwrap();
        assert_eq!(vk.name, "llama-b1-bin-win-vulkan-x64.zip");
        assert_eq!(vk.sha256, Some("ab".repeat(32)));
        assert_eq!(
            select(&rs, Variant::Cpu, WIN).unwrap().name,
            "llama-b1-bin-win-cpu-x64.zip"
        );
        assert_eq!(
            select(&rs, Variant::Hip, WIN).unwrap().name,
            "llama-b1-bin-win-rocm-10.0-x64.zip"
        );
        let arm = Target {
            os: "win",
            arch: "arm64",
        };
        assert_eq!(
            select(&rs, Variant::Vulkan, arm).unwrap().name,
            "llama-b1-bin-win-vulkan-arm64.zip"
        );
        assert!(select(&rs, Variant::Hip, arm).is_none());
    }

    #[test]
    fn skips_the_empty_latest_release_and_drafts() {
        let mut draft = release("b3", &[("llama-b3-bin-win-cpu-x64.zip", None)]);
        draft.draft = true;
        let rs = [
            release("v0.9.0", &[("nightly-tag.txt", None)]),
            draft,
            release("b2", &[("llama-b2-bin-win-cpu-x64.zip", None)]),
        ];
        assert_eq!(select(&rs, Variant::Cpu, WIN).unwrap().tag, "b2");
    }

    #[test]
    fn digests_are_validated() {
        assert!(parse_digest("sha256:abc").is_none());
        assert!(parse_digest(&format!("md5:{}", "a".repeat(32))).is_none());
        assert!(parse_digest(&format!("sha256:{}", "g".repeat(64))).is_none());
        assert!(parse_digest(&format!("sha256:{}", "a".repeat(64))).is_some());
    }
}
