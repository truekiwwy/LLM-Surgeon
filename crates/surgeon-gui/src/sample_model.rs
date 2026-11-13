//! The optional sample model of the first-run wizard: Qwen2.5-0.5B-Instruct from its
//! official Hugging Face repository, at a pinned commit so that every hash below is stable.
//! The `model.safetensors` SHA-256 is the LFS object id Hugging Face publishes; the small
//! files were hashed from the same commit. Size and license are shown before any download.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use crate::net::{Fetcher, NetError};

pub struct ModelFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub struct SampleModel {
    pub id: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub license: &'static str,
    pub files: &'static [ModelFile],
}

pub const QWEN: SampleModel = SampleModel {
    id: "Qwen2.5-0.5B-Instruct",
    repo: "Qwen/Qwen2.5-0.5B-Instruct",
    revision: "7ae557604adf67be50417f59c2c2f167def9a775",
    license: "Apache-2.0",
    files: &[
        ModelFile {
            name: "model.safetensors",
            size: 988_097_824,
            sha256: "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        },
        ModelFile {
            name: "tokenizer.json",
            size: 7_031_645,
            sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        },
        ModelFile {
            name: "tokenizer_config.json",
            size: 7_305,
            sha256: "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        },
        ModelFile {
            name: "config.json",
            size: 659,
            sha256: "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        },
        ModelFile {
            name: "generation_config.json",
            size: 242,
            sha256: "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
        },
        ModelFile {
            name: "LICENSE",
            size: 11_343,
            sha256: "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        },
    ],
};

impl SampleModel {
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// The official download URL of one file, at the pinned revision.
    pub fn url(&self, file: &ModelFile) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, file.name
        )
    }

    /// Where the model lands inside a workspace.
    pub fn dir(&self, workspace: &Path) -> PathBuf {
        workspace.join("models").join(self.id)
    }

    /// Every file is there with the right size (the hash was checked when it was written).
    pub fn is_installed(&self, workspace: &Path) -> bool {
        let dir = self.dir(workspace);
        self.files
            .iter()
            .all(|f| fs::metadata(dir.join(f.name)).is_ok_and(|m| m.len() == f.size))
    }
}

#[derive(Debug)]
pub enum ModelError {
    Net(NetError),
    /// A file does not match the pinned hash; it was deleted.
    HashMismatch {
        file: &'static str,
        actual: String,
    },
    Io(String),
    Cancelled,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Net(e) => e.fmt(f),
            Self::HashMismatch { file, actual } => {
                write!(
                    f,
                    "SHA-256 incorrect pour {file} ({actual}), fichier supprimé"
                )
            }
            Self::Io(e) => f.write_str(e),
            Self::Cancelled => f.write_str("téléchargement annulé"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Downloads and verifies every file into `dest`; files already complete are skipped, so a
/// rerun after an interruption or a repair only fetches what is missing. `progress` gets
/// `(file index, bytes done of that file)`.
pub fn install(
    fetcher: &Fetcher,
    model: &SampleModel,
    url_of: &dyn Fn(&ModelFile) -> String,
    dest: &Path,
    cancel: &AtomicBool,
    mut progress: impl FnMut(usize, u64),
) -> Result<(), ModelError> {
    fs::create_dir_all(dest).map_err(|e| ModelError::Io(format!("{} : {e}", dest.display())))?;
    for (index, file) in model.files.iter().enumerate() {
        let target = dest.join(file.name);
        let complete = crate::net::hash_file(&target).is_ok_and(|h| h == file.sha256);
        if complete {
            continue;
        }
        let part = dest.join(format!("{}.part", file.name));
        let sha = fetcher
            .download(&url_of(file), &part, Some(file.size), cancel, |done, _| {
                progress(index, done)
            })
            .map_err(|e| match e {
                NetError::Cancelled => ModelError::Cancelled,
                other => ModelError::Net(other),
            })?;
        if sha != file.sha256 {
            let _ = fs::remove_file(&part);
            return Err(ModelError::HashMismatch {
                file: file.name,
                actual: sha,
            });
        }
        fs::rename(&part, &target)
            .map_err(|e| ModelError::Io(format!("{} : {e}", target.display())))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::Policy;
    use crate::testkit::FakeReleases;
    use sha2::{Digest, Sha256};

    fn leak_model(name: &'static str, data: &[u8], sha: Option<String>) -> &'static SampleModel {
        let sha = sha.unwrap_or_else(|| {
            Sha256::digest(data)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        });
        let files: &'static [ModelFile] = Box::leak(Box::new([ModelFile {
            name,
            size: data.len() as u64,
            sha256: Box::leak(sha.into_boxed_str()),
        }]));
        Box::leak(Box::new(SampleModel {
            id: "tiny",
            repo: "x/y",
            revision: "r",
            license: "MIT",
            files,
        }))
    }

    #[test]
    fn the_pinned_manifest_is_consistent() {
        assert_eq!(QWEN.files.len(), 6);
        assert!(QWEN.files.iter().all(|f| f.sha256.len() == 64));
        assert_eq!(QWEN.license, "Apache-2.0");
        assert!(QWEN.total_size() > 988_000_000);
        let url = QWEN.url(&QWEN.files[3]);
        assert!(
            url.starts_with("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae5576")
        );
        assert!(crate::net::Policy::production().allows(&url));
    }

    #[test]
    fn files_are_verified_then_skipped_on_rerun() {
        let data = b"hello model".to_vec();
        let fake = FakeReleases::serve("f.bin", data.clone(), false);
        let model = leak_model("f.bin", &data, None);
        let base = fake.api_url().replace("/releases", "/a");
        let url_of = move |f: &ModelFile| format!("{base}/{}", f.name);
        let fetcher = Fetcher::new(Policy::loopback());
        let dir = tempfile::tempdir().unwrap();
        install(
            &fetcher,
            model,
            &url_of,
            dir.path(),
            &AtomicBool::new(false),
            |_, _| {},
        )
        .unwrap();
        assert!(
            model.is_installed(dir.path()) || fs::read(dir.path().join("f.bin")).unwrap() == data
        );
        let hits = fake.archive_hits();
        install(
            &fetcher,
            model,
            &url_of,
            dir.path(),
            &AtomicBool::new(false),
            |_, _| {},
        )
        .unwrap();
        assert_eq!(
            fake.archive_hits(),
            hits,
            "a complete file is not fetched again"
        );
    }

    #[test]
    fn a_wrong_hash_deletes_the_file() {
        let data = b"hello model".to_vec();
        let fake = FakeReleases::serve("f.bin", data.clone(), false);
        let model = leak_model("f.bin", &data, Some("0".repeat(64)));
        let base = fake.api_url().replace("/releases", "/a");
        let url_of = move |f: &ModelFile| format!("{base}/{}", f.name);
        let dir = tempfile::tempdir().unwrap();
        let err = install(
            &Fetcher::new(Policy::loopback()),
            model,
            &url_of,
            dir.path(),
            &AtomicBool::new(false),
            |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(err, ModelError::HashMismatch { .. }), "{err}");
        assert!(!dir.path().join("f.bin").exists() && !dir.path().join("f.bin.part").exists());
    }
}
