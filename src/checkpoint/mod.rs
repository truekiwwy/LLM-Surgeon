pub mod pareto;
pub mod sweep;

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::kernel::file_identity;
use crate::pipeline::job::{now_secs, read_json, sha256_file, write_json};

pub const STORE_DIR: &str = "checkpoints";
pub const MARKER: &str = ".surgeon-checkpoints";
const MANIFEST: &str = "manifest.json";
const HEADS: &str = "heads.json";
const MODEL_DIR: &str = "model";
const PARTIAL: &str = ".partial";
const STAGING: &str = ".ws-restore";
const RETIRED: &str = ".ws-retired";
const UNSHARE: &str = ".ws-unshare";

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Corrupt(String),
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Failed(String),
    #[error("annulé")]
    Cancelled,
}

impl CheckpointError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "invalid_input",
            Self::Corrupt(_) => "checksum_mismatch",
            Self::Io(_) => "io",
            Self::Failed(_) => "execution_failed",
            Self::Cancelled => "cancelled",
        }
    }
}

pub type Result<T> = std::result::Result<T, CheckpointError>;

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> CheckpointError + '_ {
    move |e| CheckpointError::Io(format!("{} : {e}", path.display()))
}

fn pipeline_error(e: crate::pipeline::PipelineError) -> CheckpointError {
    CheckpointError::Io(e.to_string())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub linked: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub label: String,
    pub parent: Option<String>,
    pub model: PathBuf,
    pub created_unix: u64,
    #[serde(default)]
    pub single_file: bool,
    pub sha256: String,
    pub files: Vec<FileEntry>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
}

impl Manifest {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Listed {
    #[serde(flatten)]
    pub manifest: Manifest,
    pub children: Vec<String>,
    pub exclusive_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Listing {
    pub store: PathBuf,
    pub checkpoints: Vec<Listed>,
    pub heads: BTreeMap<String, String>,
    pub logical_bytes: u64,
    pub disk_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Restored {
    pub id: String,
    pub model: PathBuf,
    pub sha256: String,
    pub files: usize,
    pub linked: bool,
}

fn is_backup_or_temp(name: &str) -> bool {
    crate::patch::is_backup_name(name) || name.contains(".ws-")
}

fn snapshot_files(model: &Path) -> Result<Vec<(String, PathBuf)>> {
    if model.is_file() {
        let name = model
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or_else(|| {
                CheckpointError::Invalid(format!("{} : nom de fichier", model.display()))
            })?;
        return Ok(vec![(name, model.to_path_buf())]);
    }
    let files: Vec<_> = crate::pipeline::job::files(model)
        .map_err(pipeline_error)?
        .into_iter()
        .filter(|(name, _)| !is_backup_or_temp(name))
        .collect();
    if files.is_empty() {
        return Err(CheckpointError::Invalid(format!(
            "{} : aucun fichier de modèle",
            model.display()
        )));
    }
    Ok(files)
}

fn combined_sha256(files: &[FileEntry]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for f in files {
        hasher.update(f.name.as_bytes());
        hasher.update([0]);
        hasher.update(f.sha256.as_bytes());
        hasher.update(b"\n");
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn link_or_copy(from: &Path, to: &Path) -> Result<bool> {
    match fs::hard_link(from, to) {
        Ok(()) => Ok(true),
        Err(_) => fs::copy(from, to).map(|_| false).map_err(io_error(to)),
    }
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

pub fn unshare(path: &Path) -> io::Result<bool> {
    if file_identity(path)?.links <= 1 {
        return Ok(false);
    }
    let tmp = with_suffix(path, UNSHARE);
    let copied = fs::copy(path, &tmp)
        .and_then(|_| fs::OpenOptions::new().write(true).open(&tmp)?.sync_all())
        .and_then(|()| fs::rename(&tmp, path));
    if copied.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    copied.map(|()| true)
}

fn canonical(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn key(model: &Path) -> String {
    canonical(model).to_string_lossy().into_owned()
}

fn valid_id(id: &str) -> Result<()> {
    match !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        true => Ok(()),
        false => Err(CheckpointError::Invalid(format!(
            "id « {id} » : identifiant de checkpoint invalide"
        ))),
    }
}

#[derive(Debug, Clone)]
pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn in_work(work: &Path) -> Self {
        Self {
            dir: work.join(STORE_DIR),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn prepare(&self) -> Result<()> {
        fs::create_dir_all(&self.dir).map_err(io_error(&self.dir))?;
        let marker = self.dir.join(MARKER);
        if !marker.exists() {
            fs::write(
                &marker,
                b"checkpoints: written by surgeon_checkpoint only\n",
            )
            .map_err(io_error(&marker))?;
        }
        Ok(())
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id).join(MODEL_DIR)
    }

    fn guard_model(&self, model: &Path) -> Result<()> {
        if canonical(model).starts_with(canonical(&self.dir)) {
            return Err(CheckpointError::Invalid(format!(
                "{} : un checkpoint n'est pas une copie de travail",
                model.display()
            )));
        }
        Ok(())
    }

    pub fn heads(&self) -> Result<BTreeMap<String, String>> {
        let path = self.dir.join(HEADS);
        match path.exists() {
            true => read_json(&path).map_err(pipeline_error),
            false => Ok(BTreeMap::new()),
        }
    }

    fn set_head(&self, model: &Path, id: &str) -> Result<()> {
        let mut heads = self.heads()?;
        heads.insert(key(model), id.to_owned());
        write_json(&self.dir.join(HEADS), &heads).map_err(pipeline_error)
    }

    pub fn head(&self, model: &Path) -> Result<Option<String>> {
        Ok(self.heads()?.remove(&key(model)))
    }

    fn next_id(&self) -> Result<String> {
        let mut last = 0u32;
        if self.dir.is_dir() {
            for entry in fs::read_dir(&self.dir).map_err(io_error(&self.dir))? {
                let name = entry.map_err(io_error(&self.dir))?.file_name();
                let name = name.to_string_lossy();
                let digits = name.split('.').next().unwrap_or_default();
                if let Ok(n) = digits.parse::<u32>() {
                    last = last.max(n);
                }
            }
        }
        Ok(format!("{:04}", last + 1))
    }

    pub fn manifest(&self, id: &str) -> Result<Manifest> {
        valid_id(id)?;
        let path = self.dir.join(id).join(MANIFEST);
        if !path.is_file() {
            return Err(CheckpointError::Invalid(format!(
                "checkpoint {id} introuvable"
            )));
        }
        read_json(&path).map_err(pipeline_error)
    }

    pub fn create(
        &self,
        model: &Path,
        label: &str,
        metrics: BTreeMap<String, f64>,
    ) -> Result<Manifest> {
        self.guard_model(model)?;
        let sources = snapshot_files(model)?;
        self.prepare()?;
        let id = self.next_id()?;
        let partial = self.dir.join(format!("{id}{PARTIAL}"));
        let staged = partial.join(MODEL_DIR);
        fs::create_dir_all(&staged).map_err(io_error(&staged))?;
        let built = (|| {
            let mut files = Vec::with_capacity(sources.len());
            for (name, source) in &sources {
                let target = staged.join(name);
                let linked = link_or_copy(source, &target)?;
                files.push(FileEntry {
                    name: name.clone(),
                    bytes: fs::metadata(&target).map_err(io_error(&target))?.len(),
                    sha256: sha256_file(&target).map_err(pipeline_error)?,
                    linked,
                });
            }
            let manifest = Manifest {
                id: id.clone(),
                label: label.trim().to_owned(),
                parent: self.head(model)?,
                model: canonical(model),
                created_unix: now_secs(),
                single_file: model.is_file(),
                sha256: combined_sha256(&files),
                files,
                metrics,
            };
            write_json(&partial.join(MANIFEST), &manifest).map_err(pipeline_error)?;
            let done = self.dir.join(&id);
            fs::rename(&partial, &done).map_err(io_error(&done))?;
            Ok(manifest)
        })();
        match built {
            Ok(manifest) => {
                self.set_head(model, &manifest.id)?;
                Ok(manifest)
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&partial);
                Err(e)
            }
        }
    }

    pub fn list(&self) -> Result<Listing> {
        let mut manifests = Vec::new();
        if self.dir.is_dir() {
            for entry in fs::read_dir(&self.dir).map_err(io_error(&self.dir))? {
                let entry = entry.map_err(io_error(&self.dir))?;
                let path = entry.path().join(MANIFEST);
                if path.is_file() {
                    manifests.push(read_json::<Manifest>(&path).map_err(pipeline_error)?);
                }
            }
        }
        manifests.sort_by(|a, b| a.id.cmp(&b.id));
        let mut owners: HashMap<(u64, u64), (u64, Vec<usize>)> = HashMap::new();
        for (i, m) in manifests.iter().enumerate() {
            for f in &m.files {
                let path = self.model_dir(&m.id).join(&f.name);
                let identity = file_identity(&path).map_err(io_error(&path))?;
                let owner = owners
                    .entry(identity.key())
                    .or_insert((f.bytes, Vec::new()));
                if !owner.1.contains(&i) {
                    owner.1.push(i);
                }
            }
        }
        let mut exclusive = vec![0u64; manifests.len()];
        for (bytes, users) in owners.values() {
            if let [only] = users.as_slice() {
                exclusive[*only] += bytes;
            }
        }
        let checkpoints = manifests
            .iter()
            .zip(exclusive)
            .map(|(m, exclusive_bytes)| Listed {
                manifest: m.clone(),
                children: manifests
                    .iter()
                    .filter(|c| c.parent.as_deref() == Some(m.id.as_str()))
                    .map(|c| c.id.clone())
                    .collect(),
                exclusive_bytes,
            })
            .collect();
        Ok(Listing {
            store: self.dir.clone(),
            checkpoints,
            heads: self.heads()?,
            logical_bytes: manifests.iter().map(Manifest::bytes).sum(),
            disk_bytes: owners.values().map(|(bytes, _)| bytes).sum(),
        })
    }

    pub fn verify(&self, id: &str) -> Result<Manifest> {
        let manifest = self.manifest(id)?;
        let dir = self.model_dir(id);
        let mut damaged = Vec::new();
        for f in &manifest.files {
            let path = dir.join(&f.name);
            match path.is_file() {
                true if sha256_file(&path).map_err(pipeline_error)? == f.sha256 => {}
                _ => damaged.push(f.name.clone()),
            }
        }
        match damaged.is_empty() {
            true => Ok(manifest),
            false => Err(CheckpointError::Corrupt(format!(
                "checkpoint {id} altéré (SHA-256 différent du manifeste) : {}",
                damaged.join(", ")
            ))),
        }
    }

    pub fn restore(&self, id: &str, model: &Path) -> Result<Restored> {
        self.guard_model(model)?;
        let manifest = self.verify(id)?;
        if canonical(model) != manifest.model {
            return Err(CheckpointError::Invalid(format!(
                "checkpoint {id} : pris sur {}, pas sur {}",
                manifest.model.display(),
                model.display()
            )));
        }
        let source = self.model_dir(id);
        let linked = match manifest.single_file {
            true => swap_file(&source, &manifest, model)?,
            false => swap_dir(&source, &manifest, model)?,
        };
        self.set_head(model, id)?;
        Ok(Restored {
            id: id.to_owned(),
            model: model.to_path_buf(),
            sha256: manifest.sha256.clone(),
            files: manifest.files.len(),
            linked,
        })
    }

    /// Hashes the files of `model` again: `Corrupt` unless they are the bytes of `id`.
    pub fn check_model(&self, id: &str, model: &Path) -> Result<()> {
        let manifest = self.manifest(id)?;
        let files = snapshot_files(model)?
            .into_iter()
            .map(|(name, path)| {
                Ok(FileEntry {
                    sha256: sha256_file(&path).map_err(pipeline_error)?,
                    bytes: 0,
                    name,
                    linked: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sha256 = combined_sha256(&files);
        match sha256 == manifest.sha256 {
            true => Ok(()),
            false => Err(CheckpointError::Corrupt(format!(
                "{} : SHA-256 {sha256} au lieu de {} (checkpoint {id})",
                model.display(),
                manifest.sha256
            ))),
        }
    }
}

fn swap_file(source: &Path, manifest: &Manifest, model: &Path) -> Result<bool> {
    let [file] = manifest.files.as_slice() else {
        return Err(CheckpointError::Invalid(format!(
            "checkpoint {} : un dossier, pas un fichier",
            manifest.id
        )));
    };
    let staged = with_suffix(model, STAGING);
    let _ = fs::remove_file(&staged);
    let linked = link_or_copy(&source.join(&file.name), &staged)?;
    fs::rename(&staged, model).map_err(|e| {
        let _ = fs::remove_file(&staged);
        io_error(model)(e)
    })?;
    Ok(linked)
}

fn swap_dir(source: &Path, manifest: &Manifest, model: &Path) -> Result<bool> {
    let staged = with_suffix(model, STAGING);
    let retired = with_suffix(model, RETIRED);
    for leftover in [&staged, &retired] {
        if leftover.exists() {
            fs::remove_dir_all(leftover).map_err(io_error(leftover))?;
        }
    }
    fs::create_dir_all(&staged).map_err(io_error(&staged))?;
    let mut linked = true;
    let staging = manifest.files.iter().try_for_each(|f| {
        linked &= link_or_copy(&source.join(&f.name), &staged.join(&f.name))?;
        Ok(())
    });
    if let Err(e) = staging {
        let _ = fs::remove_dir_all(&staged);
        return Err(e);
    }
    if !model.exists() {
        fs::rename(&staged, model).map_err(io_error(model))?;
        return Ok(linked);
    }
    fs::rename(model, &retired).map_err(|e| {
        let _ = fs::remove_dir_all(&staged);
        io_error(model)(e)
    })?;
    if let Err(e) = fs::rename(&staged, model) {
        let _ = fs::rename(&retired, model);
        let _ = fs::remove_dir_all(&staged);
        return Err(io_error(model)(e));
    }
    let carried = fs::read_dir(&retired).map(|entries| {
        entries.fold(true, |all, entry| {
            let moved = entry.and_then(|entry| {
                let name = entry.file_name();
                let snapshot = entry.file_type().is_ok_and(|t| t.is_file())
                    && !is_backup_or_temp(&name.to_string_lossy());
                match snapshot {
                    true => Ok(()),
                    false => fs::rename(entry.path(), model.join(&name)),
                }
            });
            all & moved.is_ok()
        })
    });
    if carried.unwrap_or(false) {
        let _ = fs::remove_dir_all(&retired);
    }
    Ok(linked)
}

#[cfg(test)]
mod tests;
