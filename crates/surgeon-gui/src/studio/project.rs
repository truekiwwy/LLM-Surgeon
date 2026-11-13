//! A project: `<workspace>/<name>/` with its versioned manifest, its runs, artifacts,
//! working copies, chat and ledger. The source model is referenced (path + SHA-256) and
//! never written: every write goes to a working copy (`copy`).

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MANIFEST: &str = "project.surgeon.json";
/// v1 is the first schema. A newer manifest is refused (never rewritten by an older studio);
/// a v2 adds its upgrade from v1 in [`load_manifest`].
pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub name: String,
    /// Seconds since the Unix epoch.
    pub created_unix: u64,
    /// The model the project works on, read only.
    #[serde(default)]
    pub source: Option<SourceModel>,
    pub state: ProjectState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceModel {
    /// Absolute path of the model file or directory.
    pub path: PathBuf,
    /// SHA-256 of the file, or of the directory's top-level files (names and contents), as
    /// `surgeon pipeline` computes it.
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectState {
    Ready,
    Running,
    Interrupted,
    NeedsAttention,
}

/// What moves a project from one state to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// A job starts (a resumed one too).
    Start,
    /// The last running job ended: ok, cancelled, failed.
    Succeed,
    Cancel,
    Fail,
    /// Opened while the manifest still says running: the previous studio died.
    Crash,
    /// The user dismissed an interruption or a failure.
    Acknowledge,
}

impl ProjectState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ready => "Prêt",
            Self::Running => "En cours",
            Self::Interrupted => "Interrompu",
            Self::NeedsAttention => "Attention requise",
        }
    }

    pub fn next(self, t: Transition) -> Result<Self, String> {
        use ProjectState as S;
        use Transition as T;
        match (self, t) {
            (_, T::Start) => Ok(S::Running),
            (S::Running, T::Succeed | T::Cancel) => Ok(S::Ready),
            (S::Running, T::Fail) => Ok(S::NeedsAttention),
            (S::Running, T::Crash) => Ok(S::Interrupted),
            (S::Interrupted | S::NeedsAttention, T::Acknowledge) => Ok(S::Ready),
            (state, t) => Err(format!("{t:?} impossible depuis l'état {}", state.label())),
        }
    }
}

/// Reads a manifest: refuses a corrupt one (with the reason) and a newer version.
pub fn load_manifest(text: &str) -> Result<Manifest, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|e| format!("manifeste illisible : {e}"))?;
    let version = value["schema_version"]
        .as_u64()
        .ok_or("manifeste sans schema_version entier")?;
    if version > u64::from(MANIFEST_VERSION) {
        return Err(format!(
            "manifeste v{version} : ce studio lit jusqu'à v{MANIFEST_VERSION}, mettez-le à jour"
        ));
    }
    if version != u64::from(MANIFEST_VERSION) {
        return Err(format!("manifeste v{version} inconnu"));
    }
    serde_json::from_value(value).map_err(|e| format!("manifeste invalide : {e}"))
}

pub fn manifest_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Manifest)).unwrap_or_default()
}

/// A folder name that stays inside the workspace on every OS.
pub fn validate_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'));
    match ok {
        true => Ok(()),
        false => Err(format!(
            "nom de projet « {name} » : 1 à 64 lettres, chiffres, - _ ou ., sans point initial"
        )),
    }
}

/// Writes through a sibling temporary file and a rename: a crash leaves the old file or the
/// new one, never half of one.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::fs::write(&tmp, bytes)
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| format!("{} : {e}", path.display()))
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[derive(Debug, Clone)]
pub struct Project {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Project {
    /// Creates `<workspace>/<name>/` and its layout; hashes the source model (seconds per GB).
    pub fn create(workspace: &Path, name: &str, source: Option<&Path>) -> Result<Self, String> {
        validate_name(name)?;
        let source = source.map(source_model).transpose()?;
        let dir = workspace.join(name);
        if dir.exists() {
            return Err(format!("{} existe déjà", dir.display()));
        }
        for sub in ["runs", "artifacts", "copies", "chat"] {
            std::fs::create_dir_all(dir.join(sub))
                .map_err(|e| format!("{} : {e}", dir.display()))?;
        }
        let project = Self {
            dir,
            manifest: Manifest {
                schema_version: MANIFEST_VERSION,
                name: name.into(),
                created_unix: now_unix(),
                source,
                state: ProjectState::Ready,
            },
        };
        std::fs::write(project.ledger_path(), b"").map_err(|e| e.to_string())?;
        project.save()?;
        Ok(project)
    }

    pub fn open(dir: &Path) -> Result<Self, String> {
        let path = dir.join(MANIFEST);
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{} : {e}", path.display()))?;
        let manifest = load_manifest(&text).map_err(|e| format!("{} : {e}", path.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    pub fn save(&self) -> Result<(), String> {
        let text = serde_json::to_string_pretty(&self.manifest).map_err(|e| e.to_string())?;
        write_atomic(&self.dir.join(MANIFEST), text.as_bytes())
    }

    /// Applies a transition and saves the manifest.
    pub fn transition(&mut self, t: Transition) -> Result<(), String> {
        self.manifest.state = self.manifest.state.next(t)?;
        self.save()
    }

    pub fn runs_dir(&self) -> PathBuf {
        self.dir.join("runs")
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.dir.join("artifacts")
    }

    pub fn copies_dir(&self) -> PathBuf {
        self.dir.join("copies")
    }

    pub fn ledger_path(&self) -> PathBuf {
        self.dir.join("ledger.jsonl")
    }

    /// The projects of a workspace, by folder name; an unreadable manifest is listed with
    /// its error, never skipped.
    pub fn list(workspace: &Path) -> Vec<(PathBuf, Result<Manifest, String>)> {
        let Ok(entries) = std::fs::read_dir(workspace) else {
            return Vec::new();
        };
        let mut out: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join(MANIFEST).is_file())
            .map(|p| {
                let manifest = Self::open(&p).map(|project| project.manifest);
                (p, manifest)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

/// Path, SHA-256 and size of a model file or directory (its top-level files).
pub fn source_model(path: &Path) -> Result<SourceModel, String> {
    use weight_surgeon::pipeline::job;
    let path = std::fs::canonicalize(path).map_err(|e| format!("{} : {e}", path.display()))?;
    let (sha256, bytes) = match path.is_dir() {
        true => (
            job::sha256_dir(&path).map_err(|e| e.to_string())?,
            super::copy::plan(&path)?.bytes,
        ),
        false => (
            job::sha256_file(&path).map_err(|e| e.to_string())?,
            std::fs::metadata(&path).map_err(|e| e.to_string())?.len(),
        ),
    };
    Ok(SourceModel {
        path,
        sha256,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ProjectState as S;
    use Transition as T;

    #[test]
    fn the_state_machine_allows_exactly_its_transitions() {
        let all = [S::Ready, S::Running, S::Interrupted, S::NeedsAttention];
        let transitions = [
            T::Start,
            T::Succeed,
            T::Cancel,
            T::Fail,
            T::Crash,
            T::Acknowledge,
        ];
        let mut allowed = Vec::new();
        for s in all {
            for t in transitions {
                if let Ok(next) = s.next(t) {
                    allowed.push((s, t, next));
                }
            }
        }
        let expected = vec![
            (S::Ready, T::Start, S::Running),
            (S::Running, T::Start, S::Running),
            (S::Running, T::Succeed, S::Ready),
            (S::Running, T::Cancel, S::Ready),
            (S::Running, T::Fail, S::NeedsAttention),
            (S::Running, T::Crash, S::Interrupted),
            (S::Interrupted, T::Start, S::Running),
            (S::Interrupted, T::Acknowledge, S::Ready),
            (S::NeedsAttention, T::Start, S::Running),
            (S::NeedsAttention, T::Acknowledge, S::Ready),
        ];
        assert_eq!(allowed, expected);
    }

    #[test]
    fn a_project_round_trips_and_lists_with_its_layout() {
        let ws = tempfile::tempdir().unwrap();
        let model = ws.path().join("m.gguf");
        std::fs::write(&model, b"weights").unwrap();
        let p = Project::create(ws.path(), "essai-1", Some(&model)).unwrap();
        for sub in [
            "runs",
            "artifacts",
            "copies",
            "chat",
            "ledger.jsonl",
            MANIFEST,
        ] {
            assert!(p.dir.join(sub).exists(), "{sub}");
        }
        let source = p.manifest.source.clone().unwrap();
        assert_eq!(source.bytes, 7);
        assert_eq!(source.sha256.len(), 64);
        assert_eq!(Project::open(&p.dir).unwrap().manifest, p.manifest);
        assert!(
            Project::create(ws.path(), "essai-1", None).is_err(),
            "exists"
        );
        let listed = Project::list(ws.path());
        assert_eq!(listed.len(), 1);
        assert!(listed[0].1.is_ok());
    }

    #[test]
    fn names_that_leave_the_workspace_are_refused() {
        for bad in ["", "..", ".hidden", "a/b", "a\\b", "c:x", &"x".repeat(65)] {
            assert!(validate_name(bad).is_err(), "{bad}");
        }
        validate_name("Qwen2.5_abl-1").unwrap();
    }

    #[test]
    fn corrupt_newer_and_unknown_manifests_are_refused_and_left_as_they_are() {
        let ws = tempfile::tempdir().unwrap();
        let p = Project::create(ws.path(), "p", None).unwrap();
        let path = p.dir.join(MANIFEST);
        for (text, needle) in [
            ("{\"schema_version\":1,\"name\":", "illisible"),
            ("{\"name\":\"p\"}", "schema_version"),
            ("{\"schema_version\":2,\"name\":\"p\"}", "v2"),
            ("{\"schema_version\":0}", "v0"),
            (
                "{\"schema_version\":1,\"name\":\"p\",\"created_unix\":1,\"state\":\"ready\",\"x\":1}",
                "invalide",
            ),
        ] {
            std::fs::write(&path, text).unwrap();
            let e = Project::open(&p.dir).unwrap_err();
            assert!(e.contains(needle), "{e}");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "untouched");
            let listed = Project::list(ws.path());
            assert!(listed[0].1.is_err(), "listed with its error");
        }
    }

    #[test]
    fn the_documented_schema_is_the_generated_one() {
        let doc = include_str!("../../../../docs/schemas/project.surgeon.schema.json");
        let doc: Value = serde_json::from_str(doc).unwrap();
        assert_eq!(
            doc,
            manifest_schema(),
            "regenerate docs/schemas/project.surgeon.schema.json"
        );
    }
}
