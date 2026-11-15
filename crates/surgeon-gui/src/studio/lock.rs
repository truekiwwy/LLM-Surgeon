//! The kernel lock on the source model: while its project is open, the studio holds every
//! file and folder of the source open for reading, sharing reading only. Windows then refuses
//! any other open that would write, truncate, rename or delete them, whoever asks: a CLI
//! runtime with its own terminal and file tools (OpenCode, Antigravity), a script, Explorer.
//! Reads stay free (the engine hashes and copies the source). No approved action writes the
//! source (the policy forbids it), so the lock lasts until the project is closed.

use std::fs::File;
use std::path::Path;

/// The handles that keep the source model read-only; dropping it releases the lock.
#[derive(Debug)]
pub struct SourceLock {
    /// Held for their effect only.
    _handles: Vec<File>,
    /// Files locked (folders not counted).
    pub files: usize,
}

/// Locks `source`, a weights file or a model folder (every file and folder under it).
pub fn lock(source: &Path) -> Result<SourceLock, String> {
    let mut handles = Vec::new();
    let mut files = 0;
    let mut pending = vec![source.to_path_buf()];
    while let Some(path) = pending.pop() {
        let dir = path.is_dir();
        if dir {
            let entries =
                std::fs::read_dir(&path).map_err(|e| format!("{} : {e}", path.display()))?;
            for entry in entries {
                pending.push(entry.map_err(|e| e.to_string())?.path());
            }
        } else {
            files += 1;
        }
        handles
            .push(open_shared_read(&path, dir).map_err(|e| format!("{} : {e}", path.display()))?);
    }
    Ok(SourceLock {
        _handles: handles,
        files,
    })
}

#[cfg(windows)]
fn open_shared_read(path: &Path, dir: bool) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x1;
    // A folder opens only with backup semantics; it then cannot be renamed or deleted.
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    let mut options = std::fs::OpenOptions::new();
    options.read(true).share_mode(FILE_SHARE_READ);
    if dir {
        options.custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
    }
    options.open(path)
}

// shortcut: Unix has advisory locks only, which other processes are free to ignore: the lock
// reports itself absent rather than pretend. Upgrade to a read-only bind mount if Linux ships.
#[cfg(not(windows))]
fn open_shared_read(_path: &Path, _dir: bool) -> std::io::Result<File> {
    Err(std::io::Error::other(
        "verrou noyau disponible sous Windows seulement",
    ))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn a_locked_source_reads_but_never_writes_moves_or_disappears() {
        let ws = tempfile::tempdir().unwrap();
        let model = ws.path().join("model");
        std::fs::create_dir_all(model.join("shards")).unwrap();
        let weights = model.join("shards").join("a.safetensors");
        std::fs::write(&weights, b"weights").unwrap();
        let held = lock(&model).unwrap();
        assert_eq!(held.files, 1);

        assert_eq!(
            std::fs::read(&weights).unwrap(),
            b"weights",
            "reads stay free"
        );
        let write = std::fs::OpenOptions::new().write(true).open(&weights);
        assert!(write.is_err(), "no other writer");
        assert!(std::fs::remove_file(&weights).is_err(), "no deletion");
        assert!(
            std::fs::rename(&model, ws.path().join("moved")).is_err(),
            "the folder stays where it is"
        );
        assert_eq!(std::fs::read(&weights).unwrap(), b"weights");

        drop(held);
        std::fs::write(&weights, b"changed").unwrap();
        assert_eq!(
            std::fs::read(&weights).unwrap(),
            b"changed",
            "released on drop"
        );
    }
}
