//! Path confinement: every path an agent sends is resolved by the OS (symlinks, `..`) and
//! must land under one of the directories the operator allowed.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Sandbox {
    roots: Vec<PathBuf>,
}

/// Names the offending argument only: echoing the resolved path would leak where a
/// symlink points, or whether something exists outside the sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError {
    pub argument: String,
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "« {} » : chemin hors des dossiers autorisés (--allow-dir)",
            self.argument
        )
    }
}

impl Sandbox {
    /// Every root must be an existing directory; it is canonicalized once here.
    pub fn new(dirs: &[PathBuf]) -> io::Result<Self> {
        if dirs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "aucun dossier autorisé",
            ));
        }
        let roots = dirs
            .iter()
            .map(|dir| {
                let root = std::fs::canonicalize(dir)
                    .map_err(|e| io::Error::new(e.kind(), format!("{} : {e}", dir.display())))?;
                match root.is_dir() {
                    true => Ok(root),
                    false => Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("{} n'est pas un dossier", dir.display()),
                    )),
                }
            })
            .collect::<io::Result<_>>()?;
        Ok(Self { roots })
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Base of relative paths when the caller gives none: the first allowed directory.
    pub fn base(&self) -> &Path {
        &self.roots[0]
    }

    /// Absolute, resolved path inside the sandbox. A path that does not exist yet (an
    /// export target, a backup directory) is accepted when its existing ancestor is inside.
    pub fn resolve(
        &self,
        argument: &str,
        path: &Path,
        base: &Path,
    ) -> Result<PathBuf, SandboxError> {
        let refused = || SandboxError {
            argument: argument.to_owned(),
        };
        let joined = match path.is_absolute() {
            true => path.to_path_buf(),
            false => base.join(path),
        };
        let resolved = resolve_lenient(&joined).ok_or_else(refused)?;
        match self.roots.iter().any(|root| within(&resolved, root)) {
            true => Ok(simplify(resolved)),
            false => Err(refused()),
        }
    }
}

/// Canonicalizes the longest existing ancestor and appends the missing names. A missing
/// tail ending in `..` is refused, and so is a dangling symlink: following it on write
/// would create a file wherever it points.
fn resolve_lenient(path: &Path) -> Option<PathBuf> {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        if let Ok(found) = std::fs::canonicalize(current) {
            let mut resolved = found;
            resolved.extend(missing.iter().rev());
            return Some(resolved);
        }
        if std::fs::symlink_metadata(current).is_ok() {
            return None;
        }
        missing.push(current.file_name()?.to_os_string());
        current = current.parent().filter(|p| !p.as_os_str().is_empty())?;
    }
}

fn within(path: &Path, root: &Path) -> bool {
    let mut components = path.components();
    root.components().all(|r| {
        components
            .next()
            .is_some_and(|c| same_component(c.as_os_str(), r.as_os_str()))
    })
}

/// NTFS is case-insensitive (ASCII folding here, enough for drive letters and names).
#[cfg(windows)]
fn same_component(a: &std::ffi::OsStr, b: &std::ffi::OsStr) -> bool {
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

#[cfg(not(windows))]
fn same_component(a: &std::ffi::OsStr, b: &std::ffi::OsStr) -> bool {
    a == b
}

/// `canonicalize` returns verbatim `\\?\C:\...` paths on Windows; external tools (Python,
/// llama.cpp) handle plain `C:\...` better. Only short drive paths are simplified.
#[cfg(windows)]
fn simplify(path: PathBuf) -> PathBuf {
    let plain = path
        .to_str()
        .and_then(|s| s.strip_prefix(r"\\?\"))
        .filter(|rest| {
            let b = rest.as_bytes();
            b.len() >= 3
                && b[0].is_ascii_alphabetic()
                && b[1] == b':'
                && b[2] == b'\\'
                && b.len() < 260
        });
    plain.map_or(path.clone(), PathBuf::from)
}

#[cfg(not(windows))]
fn simplify(path: PathBuf) -> PathBuf {
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox() -> (tempfile::TempDir, tempfile::TempDir, Sandbox) {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(allowed.path().join("m.safetensors"), "x").unwrap();
        std::fs::write(outside.path().join("secret.safetensors"), "x").unwrap();
        let sb = Sandbox::new(&[allowed.path().to_path_buf()]).unwrap();
        (allowed, outside, sb)
    }

    #[test]
    fn inside_paths_resolve_and_outside_paths_are_refused() {
        let (allowed, outside, sb) = sandbox();
        let base = sb.base().to_path_buf();
        let ok = sb
            .resolve("model", Path::new("m.safetensors"), &base)
            .unwrap();
        assert!(ok.is_absolute() && ok.ends_with("m.safetensors"));
        let nested = sb
            .resolve("backup_dir", Path::new("out/sub"), &base)
            .unwrap();
        assert!(
            nested.ends_with("out/sub"),
            "missing dirs anchored inside are allowed"
        );
        let new_file = sb
            .resolve("export_gguf", Path::new("new.gguf"), &base)
            .unwrap();
        assert!(new_file.ends_with("new.gguf"));

        let escape = format!(
            "../{}/secret.safetensors",
            outside.path().file_name().unwrap().to_string_lossy()
        );
        for path in [
            PathBuf::from(&escape),
            outside.path().join("secret.safetensors"),
            outside.path().join("absent.gguf"),
            PathBuf::from("m.safetensors/../../x"),
        ] {
            let err = sb.resolve("model", &path, &base).unwrap_err();
            assert_eq!(err.argument, "model");
            assert!(!err
                .to_string()
                .contains(&outside.path().display().to_string()));
        }
        let _ = allowed;
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_pointing_outside_are_refused() {
        let (allowed, outside, sb) = sandbox();
        std::os::unix::fs::symlink(outside.path(), allowed.path().join("link")).unwrap();
        std::os::unix::fs::symlink(outside.path().join("nope"), allowed.path().join("dangling"))
            .unwrap();
        let base = sb.base().to_path_buf();
        assert!(sb
            .resolve("model", Path::new("link/secret.safetensors"), &base)
            .is_err());
        assert!(sb
            .resolve("export_gguf", Path::new("dangling"), &base)
            .is_err());
    }

    #[cfg(windows)]
    #[test]
    fn verbatim_prefix_and_case_are_handled() {
        let (allowed, outside, sb) = sandbox();
        let base = sb.base().to_path_buf();
        let verbatim = PathBuf::from(format!(
            r"\\?\{}",
            allowed.path().join("m.safetensors").display()
        ));
        let ok = sb.resolve("model", &verbatim, &base).unwrap();
        assert!(
            !ok.to_string_lossy().starts_with(r"\\?\"),
            "simplified for external tools"
        );
        let upper = PathBuf::from(
            allowed
                .path()
                .join("M.SAFETENSORS")
                .to_string_lossy()
                .to_uppercase(),
        );
        assert!(
            sb.resolve("model", &upper, &base).is_ok(),
            "NTFS ignores case"
        );
        let outside_verbatim = PathBuf::from(format!(
            r"\\?\{}",
            outside.path().join("secret.safetensors").display()
        ));
        assert!(sb.resolve("model", &outside_verbatim, &base).is_err());
    }
}
