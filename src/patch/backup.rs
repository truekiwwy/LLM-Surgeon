//! Backup state machine: `Created -> Committed | RolledBack`.
//!
//! - `create` copies the original next to it (or into `backup_dir`) and only succeeds once
//!   the copy's SHA-256 matches the original; a failed copy is deleted.
//! - `rollback` first checks the backup's own hash (never restore a corrupted copy), then
//!   restores atomically (copy to a sibling temp file + `rename`). When `rename` is refused,
//!   typically on Windows while the engine still maps the file, it falls back to an
//!   in-place overwrite. Either way the final SHA-256 is checked against the original's.
//! - `Drop` in state `Created` (error path, early return, panic unwinding) performs a
//!   best-effort rollback and reports to stderr.
//!
//! Limit: free disk space is not checked, `std::fs` cannot query it. A full disk shows up
//! as a failed copy, which is cleaned up and reported before anything is modified.

use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::error::{PatchError, Result};
use super::executor::RollbackHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Created,
    Committed,
    RolledBack,
}

/// Not `Clone`: two guards for one backup would both restore on drop.
#[derive(Debug)]
pub struct BackupGuard {
    original: PathBuf,
    backup: PathBuf,
    sha256_original: String,
    state: State,
}

impl BackupGuard {
    pub fn create(original: &Path, backup_dir: Option<&Path>) -> Result<Self> {
        let name = original.file_name().ok_or_else(|| {
            PatchError::Backup(format!(
                "chemin sans nom de fichier : {}",
                original.display()
            ))
        })?;
        let dir = match backup_dir {
            Some(dir) if !dir.is_dir() => {
                return Err(PatchError::Backup(format!(
                    "dossier de backup inexistant : {}",
                    dir.display()
                )))
            }
            Some(dir) => dir.to_path_buf(),
            None => original.parent().map(Path::to_path_buf).unwrap_or_default(),
        };
        let sha256_original = sha256_file(original)?;
        let (backup, mut dst) = create_unique(&dir, name)?;

        let copied = (|| -> Result<()> {
            let mut src = File::open(original).map_err(|e| PatchError::io(original, &e))?;
            io::copy(&mut src, &mut dst).map_err(|e| PatchError::io(&backup, &e))?;
            dst.sync_all().map_err(|e| PatchError::io(&backup, &e))?;
            if sha256_file(&backup)? != sha256_original {
                return Err(PatchError::Backup(
                    "SHA-256 de la copie différent de l'original (fichier modifié pendant la copie ?)".into(),
                ));
            }
            Ok(())
        })();
        if let Err(e) = copied {
            drop(dst);
            let _ = fs::remove_file(&backup);
            return Err(e);
        }
        Ok(Self {
            original: original.to_path_buf(),
            backup,
            sha256_original,
            state: State::Created,
        })
    }

    pub fn backup_path(&self) -> &Path {
        &self.backup
    }

    pub fn original_path(&self) -> &Path {
        &self.original
    }

    pub fn rollback(&mut self) -> Result<()> {
        if self.state == State::Committed {
            return Err(PatchError::Backup(
                "backup déjà validé (commit), rollback refusé".into(),
            ));
        }
        if sha256_file(&self.backup)? != self.sha256_original {
            return Err(PatchError::Backup(format!(
                "backup corrompu, original laissé tel quel : {}",
                self.backup.display()
            )));
        }
        restore(&self.backup, &self.original).map_err(|e| PatchError::io(&self.original, &e))?;
        if sha256_file(&self.original)? != self.sha256_original {
            return Err(PatchError::Backup(format!(
                "SHA-256 après restauration différent : {}",
                self.original.display()
            )));
        }
        self.state = State::RolledBack;
        Ok(())
    }

    /// Keeps the patched original. The backup file stays on disk as a manual undo.
    pub fn commit(&mut self) {
        self.state = State::Committed;
    }

    /// Disarms the guard and deletes its backup (nothing was modified yet).
    fn discard(mut self) {
        self.state = State::Committed;
        let _ = fs::remove_file(&self.backup);
    }
}

impl Drop for BackupGuard {
    fn drop(&mut self) {
        if self.state != State::Created {
            return;
        }
        match self.rollback() {
            Ok(()) => eprintln!(
                "weight-surgeon : opération non validée, {} restauré depuis {}",
                self.original.display(),
                self.backup.display()
            ),
            Err(e) => eprintln!(
                "weight-surgeon : CRITIQUE, restauration de {} impossible ({e}). Backup : {}",
                self.original.display(),
                self.backup.display()
            ),
        }
    }
}

impl RollbackHandle for BackupGuard {
    fn rollback(&mut self) -> Result<()> {
        BackupGuard::rollback(self)
    }
    fn commit(&mut self) {
        BackupGuard::commit(self);
    }
    fn backup_path(&self) -> Option<PathBuf> {
        Some(self.backup.clone())
    }
}

/// One guard per shard; all-or-nothing creation.
#[derive(Debug)]
pub struct BackupSet {
    guards: Vec<BackupGuard>,
}

impl BackupSet {
    /// Backups next to each shard.
    pub fn create_many(paths: &[PathBuf]) -> Result<Self> {
        Self::create_many_in(paths, None)
    }

    /// If one copy fails, the backups already made are deleted (nothing was modified).
    pub fn create_many_in(paths: &[PathBuf], backup_dir: Option<&Path>) -> Result<Self> {
        let mut guards = Vec::with_capacity(paths.len());
        for path in paths {
            match BackupGuard::create(path, backup_dir) {
                Ok(guard) => guards.push(guard),
                Err(e) => {
                    guards.into_iter().for_each(BackupGuard::discard);
                    return Err(e);
                }
            }
        }
        Ok(Self { guards })
    }

    pub fn guards(&self) -> &[BackupGuard] {
        &self.guards
    }

    /// Tries every shard even if one fails, then reports all failures.
    pub fn rollback_all(&mut self) -> Result<()> {
        let errors: Vec<String> = self
            .guards
            .iter_mut()
            .filter_map(|g| g.rollback().err().map(|e| e.to_string()))
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(PatchError::Backup(errors.join(" ; ")))
        }
    }

    pub fn commit_all(&mut self) {
        self.guards.iter_mut().for_each(BackupGuard::commit);
    }
}

impl RollbackHandle for BackupSet {
    fn rollback(&mut self) -> Result<()> {
        self.rollback_all()
    }
    fn commit(&mut self) {
        self.commit_all();
    }
    /// Directory of the first backup (shards normally share one).
    fn backup_path(&self) -> Option<PathBuf> {
        self.guards
            .first()
            .and_then(|g| g.backup.parent().map(Path::to_path_buf))
    }
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|e| PatchError::io(path, &e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).map_err(|e| PatchError::io(path, &e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `<dir>/<name>.bak-<unix secs>[-n]`, created with `create_new` so an existing backup is
/// never overwritten.
fn create_unique(dir: &Path, name: &OsStr) -> Result<(PathBuf, File)> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    for n in 0..1000u32 {
        let mut file_name = name.to_owned();
        file_name.push(format!(".bak-{stamp}"));
        if n > 0 {
            file_name.push(format!("-{n}"));
        }
        let path = dir.join(file_name);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(PatchError::io(&path, &e)),
        }
    }
    Err(PatchError::Backup(format!(
        "trop de backups existants dans {}",
        dir.display()
    )))
}

fn restore(backup: &Path, original: &Path) -> io::Result<()> {
    let mut tmp = original.as_os_str().to_owned();
    tmp.push(".ws-restore");
    let tmp = PathBuf::from(tmp);
    let atomic = copy_synced(backup, &tmp).and_then(|()| fs::rename(&tmp, original));
    if atomic.is_ok() {
        return Ok(());
    }
    let _ = fs::remove_file(&tmp);
    overwrite_in_place(backup, original)
}

fn copy_synced(src: &Path, dst: &Path) -> io::Result<()> {
    let mut out = File::create(dst)?;
    io::copy(&mut File::open(src)?, &mut out)?;
    out.sync_all()
}

/// Rewrites the content without truncating first: allowed on Windows while the file is
/// mapped, where both `rename` over it and `set_len` are refused.
fn overwrite_in_place(src: &Path, dst: &Path) -> io::Result<()> {
    let mut out = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(dst)?;
    let len = io::copy(&mut File::open(src)?, &mut out)?;
    if out.metadata()?.len() != len {
        out.set_len(len)?;
    }
    out.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_with(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn create_makes_identical_copy_next_to_original() {
        let dir = tempfile::tempdir().unwrap();
        let orig = file_with(dir.path(), "m.safetensors", b"weights-v1");
        let mut guard = BackupGuard::create(&orig, None).unwrap();
        let backup = guard.backup_path().to_path_buf();
        assert_eq!(backup.parent(), Some(dir.path()));
        assert!(backup
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("m.safetensors.bak-"));
        assert_eq!(fs::read(&backup).unwrap(), b"weights-v1");
        assert_eq!(sha256_file(&backup).unwrap(), sha256_file(&orig).unwrap());

        // A second backup in the same second gets a distinct name instead of overwriting.
        let mut other = BackupGuard::create(&orig, None).unwrap();
        assert_ne!(other.backup_path(), backup);
        guard.commit();
        other.commit();
    }

    #[test]
    fn rollback_restores_exactly_and_commit_keeps_changes() {
        let dir = tempfile::tempdir().unwrap();
        let orig = file_with(dir.path(), "m.bin", &[1, 2, 3, 4, 5]);
        let mut guard = BackupGuard::create(&orig, None).unwrap();
        fs::write(&orig, [9, 9]).unwrap();
        guard.rollback().unwrap();
        assert_eq!(fs::read(&orig).unwrap(), [1, 2, 3, 4, 5]);

        let mut guard = BackupGuard::create(&orig, None).unwrap();
        fs::write(&orig, [7; 5]).unwrap();
        guard.commit();
        assert!(matches!(guard.rollback(), Err(PatchError::Backup(_))));
        drop(guard);
        assert_eq!(fs::read(&orig).unwrap(), [7; 5]);
    }

    #[test]
    fn drop_without_commit_restores() {
        let dir = tempfile::tempdir().unwrap();
        let orig = file_with(dir.path(), "m.bin", b"original");
        {
            let _guard = BackupGuard::create(&orig, None).unwrap();
            fs::write(&orig, b"half-patched").unwrap();
        }
        assert_eq!(fs::read(&orig).unwrap(), b"original");
    }

    #[test]
    fn rollback_recreates_a_deleted_original() {
        let dir = tempfile::tempdir().unwrap();
        let orig = file_with(dir.path(), "m.bin", b"original");
        let mut guard = BackupGuard::create(&orig, None).unwrap();
        fs::remove_file(&orig).unwrap();
        guard.rollback().unwrap();
        assert_eq!(fs::read(&orig).unwrap(), b"original");
    }

    #[test]
    fn corrupted_or_missing_backup_never_touches_the_original() {
        let dir = tempfile::tempdir().unwrap();
        let orig = file_with(dir.path(), "m.bin", b"original");
        let mut guard = BackupGuard::create(&orig, None).unwrap();
        fs::write(&orig, b"patched").unwrap();
        fs::write(guard.backup_path(), b"garbage").unwrap();
        assert!(matches!(guard.rollback(), Err(PatchError::Backup(_))));
        assert_eq!(fs::read(&orig).unwrap(), b"patched");

        fs::remove_file(guard.backup_path()).unwrap();
        assert!(matches!(guard.rollback(), Err(PatchError::Io { .. })));
        guard.commit(); // silence the drop-time retry
    }

    #[test]
    fn missing_backup_dir_or_original_fails_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let orig = file_with(dir.path(), "m.bin", b"x");
        let err = BackupGuard::create(&orig, Some(&dir.path().join("nope"))).unwrap_err();
        assert!(matches!(err, PatchError::Backup(_)));
        assert!(matches!(
            BackupGuard::create(&dir.path().join("absent"), None),
            Err(PatchError::Io { .. })
        ));
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            1,
            "no partial backup left"
        );
    }

    #[test]
    fn backup_set_over_three_shards() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let shards: Vec<PathBuf> = (0..3)
            .map(|i| file_with(dir.path(), &format!("s{i}.safetensors"), &[i; 4]))
            .collect();
        let mut set = BackupSet::create_many_in(&shards, Some(backups.path())).unwrap();
        assert_eq!(set.guards().len(), 3);
        assert_eq!(
            RollbackHandle::backup_path(&set).as_deref(),
            Some(backups.path())
        );
        for s in &shards {
            fs::write(s, b"patched").unwrap();
        }
        set.rollback_all().unwrap();
        for (i, s) in shards.iter().enumerate() {
            assert_eq!(fs::read(s).unwrap(), [i as u8; 4]);
        }

        let mut set = BackupSet::create_many(&shards).unwrap();
        fs::write(&shards[1], b"kept").unwrap();
        set.commit_all();
        drop(set);
        assert_eq!(fs::read(&shards[1]).unwrap(), b"kept");

        // All-or-nothing: a missing shard leaves no stray backup behind.
        let before = fs::read_dir(backups.path()).unwrap().count();
        let mut with_missing = shards.clone();
        with_missing.push(dir.path().join("absent.safetensors"));
        assert!(BackupSet::create_many_in(&with_missing, Some(backups.path())).is_err());
        assert_eq!(fs::read_dir(backups.path()).unwrap().count(), before);
    }

    /// The engine keeps the model mapped while the patch runs. On Windows `rename` over a
    /// mapped file is refused, so this exercises the in-place fallback there.
    #[test]
    fn rollback_works_while_the_file_is_memory_mapped() {
        let dir = tempfile::tempdir().unwrap();
        let orig = file_with(dir.path(), "m.bin", &[1u8; 4096]);
        let mut guard = BackupGuard::create(&orig, None).unwrap();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&orig)
            .unwrap();
        // SAFETY: test-only mapping of a temp file nothing else touches.
        let mut map = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };
        map[..4].copy_from_slice(&[9, 9, 9, 9]);
        map.flush().unwrap();
        guard.rollback().unwrap();
        drop(map);
        assert_eq!(fs::read(&orig).unwrap(), vec![1u8; 4096]);
    }
}
