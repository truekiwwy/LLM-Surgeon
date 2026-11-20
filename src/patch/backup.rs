//! Backup state machine: `Created -> Committed | RolledBack`.
//!
//! - `create` copies the original next to it (or into `backup_dir`) and only succeeds once
//!   the copy's SHA-256 matches the original; a failed copy is deleted.
//! - Every backup of one `apply` carries the same checkpoint, `<name>.bak-<secs>-<n>`, newer
//!   than every backup already in its folder: the newest checkpoint is exactly the state
//!   before the last `apply` (`patch::restore`). When the model is already byte for byte in
//!   that state (apply, restore, apply again), the checkpoint is reused instead of copied.
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
use std::io::{self, Read};
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

/// Where the backup of `original` goes.
fn backup_folder(original: &Path, backup_dir: Option<&Path>) -> Result<PathBuf> {
    match backup_dir {
        Some(dir) if !dir.is_dir() => Err(PatchError::Backup(format!(
            "dossier de backup inexistant : {}",
            dir.display()
        ))),
        Some(dir) => Ok(dir.to_path_buf()),
        None => Ok(original.parent().map(Path::to_path_buf).unwrap_or_default()),
    }
}

impl BackupGuard {
    /// A new backup, in a checkpoint of its own.
    pub fn create(original: &Path, backup_dir: Option<&Path>) -> Result<Self> {
        let dir = backup_folder(original, backup_dir)?;
        let checkpoint = next_checkpoint(std::slice::from_ref(&dir));
        Self::create_at(original, &dir, checkpoint)
    }

    /// The backup of one file for an `apply`: see [`BackupSet::create_many_in`].
    pub fn checkpoint(original: &Path, backup_dir: Option<&Path>) -> Result<Self> {
        let mut set = BackupSet::create_many_in(&[original.to_path_buf()], backup_dir)?;
        set.guards
            .pop()
            .ok_or_else(|| PatchError::Backup("aucune sauvegarde créée".into()))
    }

    fn create_at(original: &Path, dir: &Path, checkpoint: Checkpoint) -> Result<Self> {
        let name = original.file_name().ok_or_else(|| {
            PatchError::Backup(format!(
                "chemin sans nom de fichier : {}",
                original.display()
            ))
        })?;
        let sha256_original = sha256_file(original)?;
        let (backup, mut dst) = create_unique(dir, name, checkpoint)?;

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

    /// Hex SHA-256 of the original, which the backup matches.
    pub fn original_sha256(&self) -> &str {
        &self.sha256_original
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

    /// One checkpoint for all `paths`, or the newest one when the model already is in the
    /// state it holds. If one copy fails, the backups already made are deleted (nothing was
    /// modified).
    pub fn create_many_in(paths: &[PathBuf], backup_dir: Option<&Path>) -> Result<Self> {
        let dirs = paths
            .iter()
            .map(|p| backup_folder(p, backup_dir))
            .collect::<Result<Vec<_>>>()?;
        if let Some(set) = Self::reuse_newest(paths, &dirs)? {
            return Ok(set);
        }
        let checkpoint = next_checkpoint(&dirs);
        let mut guards = Vec::with_capacity(paths.len());
        for (path, dir) in paths.iter().zip(&dirs) {
            match BackupGuard::create_at(path, dir, checkpoint) {
                Ok(guard) => guards.push(guard),
                Err(e) => {
                    guards.into_iter().for_each(BackupGuard::discard);
                    return Err(e);
                }
            }
        }
        Ok(Self { guards })
    }

    /// The guards over the newest checkpoint, if it covers `paths` and every file it holds
    /// is identical to the model now. Its other files must match too: restoring it has to
    /// give back the state before this `apply`, whatever else it holds.
    fn reuse_newest(paths: &[PathBuf], dirs: &[PathBuf]) -> Result<Option<Self>> {
        let (Some(dir), Some(model_dir)) = (dirs.first(), paths.first().and_then(|p| p.parent()))
        else {
            return Ok(None);
        };
        if dirs.iter().any(|d| d != dir) {
            return Ok(None);
        }
        let found = backups_in(dir).unwrap_or_default();
        let Some(newest) = found.iter().map(|b| b.checkpoint).max() else {
            return Ok(None);
        };
        // Older folders: the shards of an apply are not linked, nothing to rely on.
        if newest.1 == 0 {
            return Ok(None);
        }
        let group: Vec<&Backup> = found.iter().filter(|b| b.checkpoint == newest).collect();
        let covered = |p: &PathBuf| {
            let name = p.file_name().and_then(OsStr::to_str);
            group.iter().any(|b| Some(b.original.as_str()) == name)
        };
        let same_size = |b: &&Backup| {
            let size = |p: &Path| fs::metadata(p).map(|m| m.len()).ok();
            let len = size(&model_dir.join(&b.original));
            len.is_some() && len == size(&b.path)
        };
        if !paths.iter().all(covered) || !group.iter().all(same_size) {
            return Ok(None);
        }
        let mut guards = Vec::with_capacity(paths.len());
        for backup in group {
            let original = model_dir.join(&backup.original);
            let sha256_original = sha256_file(&original)?;
            if sha256_file(&backup.path)? != sha256_original {
                return Ok(None);
            }
            if paths.contains(&original) {
                guards.push(BackupGuard {
                    original,
                    backup: backup.path.clone(),
                    sha256_original,
                    state: State::Created,
                });
            }
        }
        Ok(Some(Self { guards }))
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

/// (unix secs, n) of `<name>.bak-<secs>-<n>`: the order of creation, shared by the backups of
/// one `apply`. `n = 0` (no suffix) only exists in folders written by older versions, where
/// each file had its own stamp.
pub(super) type Checkpoint = (u64, u32);

/// Whether `name` is a backup `apply` writes: its folder holds a patched model.
pub fn is_backup_name(name: &str) -> bool {
    parse_backup_name(name).is_some()
}

/// `<original>.bak-<secs>[-n]` -> (original, checkpoint).
pub(super) fn parse_backup_name(name: &str) -> Option<(&str, Checkpoint)> {
    let at = name.rfind(".bak-")?;
    let suffix = &name[at + ".bak-".len()..];
    let (secs, n) = match suffix.split_once('-') {
        Some((secs, n)) => (secs, n.parse().ok()?),
        None => (suffix, 0),
    };
    let original = &name[..at];
    (!original.is_empty()).then_some((original, (secs.parse().ok()?, n)))
}

/// A backup file found in a folder.
pub(super) struct Backup {
    pub original: String,
    pub checkpoint: Checkpoint,
    pub path: PathBuf,
}

pub(super) fn backups_in(dir: &Path) -> io::Result<Vec<Backup>> {
    Ok(fs::read_dir(dir)?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let (original, checkpoint) = parse_backup_name(name)?;
            let original = original.to_owned();
            path.is_file().then_some(Backup {
                original,
                checkpoint,
                path,
            })
        })
        .collect())
}

/// Strictly newer than every backup of `dirs`, even within one second or if the clock moved.
fn next_checkpoint(dirs: &[PathBuf]) -> Checkpoint {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let newest = dirs
        .iter()
        .flat_map(|dir| backups_in(dir).unwrap_or_default())
        .map(|b| b.checkpoint)
        .max();
    match newest {
        Some((secs, n)) if secs >= now => (secs, n + 1),
        _ => (now, 1),
    }
}

/// `<dir>/<name>.bak-<secs>-<n>`, created with `create_new` so an existing backup is never
/// overwritten.
fn create_unique(dir: &Path, name: &OsStr, (secs, n): Checkpoint) -> Result<(PathBuf, File)> {
    let mut file_name = name.to_owned();
    file_name.push(format!(".bak-{secs}-{n}"));
    let path = dir.join(file_name);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| PatchError::io(&path, &e))?;
    Ok((path, file))
}

pub(super) fn restore(backup: &Path, original: &Path) -> io::Result<()> {
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
        let name = backup.file_name().unwrap().to_str().unwrap();
        assert_eq!(parse_backup_name(name).unwrap().0, "m.safetensors");
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

    #[test]
    fn one_apply_is_one_checkpoint_and_each_is_newer() {
        let dir = tempfile::tempdir().unwrap();
        let shards: Vec<PathBuf> = (0..2)
            .map(|i| file_with(dir.path(), &format!("s{i}.safetensors"), &[i; 4]))
            .collect();
        let stamp = |set: &BackupSet, i: usize| {
            let name = set.guards()[i].backup_path().file_name().unwrap();
            parse_backup_name(name.to_str().unwrap()).unwrap().1
        };
        let mut first = BackupSet::create_many(&shards).unwrap();
        assert_eq!(stamp(&first, 0), stamp(&first, 1), "same checkpoint");
        assert!(stamp(&first, 0).1 >= 1);
        first.commit_all();
        fs::write(&shards[0], b"changed").unwrap();
        // Only one shard this time, and a checkpoint above the two files of the first.
        let mut second = BackupSet::create_many(&shards[..1]).unwrap();
        assert!(stamp(&second, 0) > stamp(&first, 0));
        second.commit_all();
    }

    #[test]
    fn the_same_state_is_not_copied_twice() {
        let dir = tempfile::tempdir().unwrap();
        let shards: Vec<PathBuf> = (0..2)
            .map(|i| file_with(dir.path(), &format!("s{i}.bin"), &[i; 4]))
            .collect();
        let backups = |d: &Path| backups_in(d).unwrap().len();
        let mut first = BackupSet::create_many(&shards).unwrap();
        first.commit_all();
        assert_eq!(backups(dir.path()), 2);
        // Patched then restored: same bytes as the checkpoint, nothing new on disk.
        let mut again = BackupSet::create_many(&shards[..1]).unwrap();
        assert_eq!(backups(dir.path()), 2);
        assert_eq!(
            again.guards()[0].backup_path(),
            first.guards()[0].backup_path()
        );
        // A failed apply on it rolls back from the reused file, which stays.
        fs::write(&shards[0], b"half").unwrap();
        again.rollback_all().unwrap();
        assert_eq!(fs::read(&shards[0]).unwrap(), [0; 4]);
        assert!(first.guards()[0].backup_path().is_file());
        // A new state, or a file of the checkpoint that moved on: a new checkpoint.
        fs::write(&shards[1], b"other").unwrap();
        let mut moved = BackupSet::create_many(&shards[..1]).unwrap();
        moved.commit_all();
        assert_eq!(backups(dir.path()), 3);
        assert_ne!(
            moved.guards()[0].backup_path(),
            first.guards()[0].backup_path()
        );
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
