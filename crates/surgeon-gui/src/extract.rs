//! Safe zip extraction: nothing leaves the target folder ("zip-slip"), no symbolic link is
//! created, and an archive that claims an absurd size is refused before it fills the disk.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use zip::ZipArchive;

/// Entry and size ceilings: a llama.cpp archive is a few hundred files and well under 1 GiB
/// (the CUDA bundles reach 600 MiB), so these only stop decompression bombs.
const MAX_ENTRIES: usize = 20_000;
const MAX_TOTAL_BYTES: u64 = 8 << 30;

#[derive(Debug)]
pub enum ExtractError {
    Archive(String),
    Unsafe(String),
    TooBig(&'static str),
    Cancelled,
    Io(String),
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Archive(e) => write!(f, "archive illisible : {e}"),
            Self::Unsafe(name) => write!(f, "entrée dangereuse refusée : « {name} »"),
            Self::TooBig(limit) => {
                write!(f, "archive trop grosse ou trop fournie (limite {limit})")
            }
            Self::Cancelled => f.write_str("extraction annulée"),
            Self::Io(e) => f.write_str(e),
        }
    }
}

impl std::error::Error for ExtractError {}

/// The relative path of an entry, or `None` when it could escape (absolute, drive letter,
/// UNC, `..`, NUL, an empty name). `\` is a separator too: Windows tools write it.
pub fn safe_relative(name: &str) -> Option<PathBuf> {
    if name.is_empty() || name.contains('\0') {
        return None;
    }
    let normalized = name.replace('\\', "/");
    let mut out = PathBuf::new();
    for part in normalized.split('/') {
        match part {
            "" | "." => continue,
            ".." => return None,
            p if p.len() >= 2
                && p.as_bytes()[1] == b':'
                && p.as_bytes()[0].is_ascii_alphabetic() =>
            {
                return None
            }
            p => out.push(p),
        }
    }
    // A leading `/` (or `\\server`) yields empty first parts above: reject it explicitly.
    let absolute = normalized.starts_with('/');
    let clean = out.components().all(|c| matches!(c, Component::Normal(_)));
    (!absolute && clean && out.components().count() > 0).then_some(out)
}

/// Extracts `archive` into `dest` (created). Returns the number of files written.
pub fn extract_zip(
    archive: &Path,
    dest: &Path,
    cancel: &AtomicBool,
) -> Result<usize, ExtractError> {
    let io_err = |e: io::Error| ExtractError::Io(format!("{} : {e}", archive.display()));
    let file = File::open(archive).map_err(io_err)?;
    let mut zip = ZipArchive::new(file).map_err(|e| ExtractError::Archive(e.to_string()))?;
    if zip.len() > MAX_ENTRIES {
        return Err(ExtractError::TooBig("20000 entrées"));
    }
    // Names and sizes are checked for the whole archive before the first byte is written.
    let mut plan = Vec::with_capacity(zip.len());
    let mut total = 0u64;
    for i in 0..zip.len() {
        let entry = zip
            .by_index(i)
            .map_err(|e| ExtractError::Archive(e.to_string()))?;
        let name = entry
            .name()
            .map_err(|e| ExtractError::Archive(e.to_string()))?
            .into_owned();
        let rel = safe_relative(&name).ok_or_else(|| ExtractError::Unsafe(name.clone()))?;
        let symlink = entry
            .unix_mode()
            .is_some_and(|m| m & 0o170_000 == 0o120_000);
        if symlink {
            return Err(ExtractError::Unsafe(name));
        }
        total = total.saturating_add(entry.size());
        plan.push((i, rel, entry.is_dir()));
    }
    if total > MAX_TOTAL_BYTES {
        return Err(ExtractError::TooBig("8 Gio"));
    }
    fs::create_dir_all(dest).map_err(|e| ExtractError::Io(format!("{} : {e}", dest.display())))?;
    let mut written = 0;
    for (i, rel, is_dir) in plan {
        if cancel.load(Ordering::Relaxed) {
            return Err(ExtractError::Cancelled);
        }
        let target = dest.join(&rel);
        let io_err = |e: io::Error| ExtractError::Io(format!("{} : {e}", target.display()));
        if is_dir {
            fs::create_dir_all(&target).map_err(io_err)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(io_err)?;
        }
        let mut entry = zip
            .by_index(i)
            .map_err(|e| ExtractError::Archive(e.to_string()))?;
        let mut out = File::create(&target).map_err(io_err)?;
        copy_cancellable(&mut entry, &mut out, cancel).map_err(|e| match e.kind() {
            io::ErrorKind::Interrupted => ExtractError::Cancelled,
            _ => io_err(e),
        })?;
        written += 1;
    }
    Ok(written)
}

fn copy_cancellable(
    from: &mut impl Read,
    to: &mut impl Write,
    cancel: &AtomicBool,
) -> io::Result<()> {
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        let n = from.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        to.write_all(&buf[..n])?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zip::write::SimpleFileOptions;

    fn make_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let mut w = zip::ZipWriter::new(File::create(path).unwrap());
        for (name, data) in entries {
            w.start_file(*name, SimpleFileOptions::default()).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap();
    }

    #[test]
    fn hostile_names_are_rejected() {
        for bad in [
            "../evil.txt",
            "a/../../evil.txt",
            "..\\evil.txt",
            "/etc/passwd",
            "\\\\server\\share\\x",
            "C:\\Windows\\x.dll",
            "c:/x",
            "",
            "a\0b",
            ".",
        ] {
            assert!(safe_relative(bad).is_none(), "{bad:?} must be refused");
        }
        assert_eq!(
            safe_relative("bin\\llama-server.exe").unwrap(),
            PathBuf::from("bin/llama-server.exe")
        );
        assert_eq!(
            safe_relative("./a//b.txt").unwrap(),
            PathBuf::from("a/b.txt")
        );
    }

    #[test]
    fn a_normal_archive_is_extracted() {
        let dir = tempfile::tempdir().unwrap();
        let zip = dir.path().join("ok.zip");
        make_zip(
            &zip,
            &[("llama-server.exe", b"MZ"), ("sub/ggml.dll", b"dll")],
        );
        let out = dir.path().join("out");
        assert_eq!(extract_zip(&zip, &out, &AtomicBool::new(false)).unwrap(), 2);
        assert_eq!(fs::read(out.join("sub").join("ggml.dll")).unwrap(), b"dll");
    }

    #[test]
    fn a_zip_slip_archive_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let zip = dir.path().join("evil.zip");
        make_zip(&zip, &[("fine.txt", b"x"), ("../escaped.txt", b"pwn")]);
        let out = dir.path().join("out");
        let err = extract_zip(&zip, &out, &AtomicBool::new(false)).unwrap_err();
        assert!(matches!(err, ExtractError::Unsafe(_)), "{err}");
        assert!(
            !out.exists(),
            "nothing may be written before the whole archive is checked"
        );
        assert!(!dir.path().join("escaped.txt").exists());
    }

    #[test]
    fn absolute_entries_and_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let zip = dir.path().join("abs.zip");
        make_zip(&zip, &[("/abs.txt", b"x")]);
        assert!(extract_zip(&zip, &dir.path().join("o"), &AtomicBool::new(false)).is_err());
        let ok = dir.path().join("ok.zip");
        make_zip(&ok, &[("a.txt", b"x")]);
        let err = extract_zip(&ok, &dir.path().join("o2"), &AtomicBool::new(true)).unwrap_err();
        assert!(matches!(err, ExtractError::Cancelled));
    }
}
