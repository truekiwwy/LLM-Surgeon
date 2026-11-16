//! `surgeon restore` / `surgeon_restore`: puts a model back byte for byte, from the backups
//! of `apply` or from the origin model. Every file is hashed before and after; the restore
//! fails unless the final SHA-256 is the source's. Sources are only read, never removed.
//!
//! Detection without `--backup`: the backups of a file are `<name>.bak-<secs>[-n]` (see
//! `backup::create_unique`), the newest one is the state before the last `apply`. In a
//! sharded folder each `apply` backs up only the files it touches, so when one file has
//! several backups nothing tells which ones belong together: the detection refuses rather
//! than mix two states.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use serde::Serialize;
use serde_json::json;

use super::backup::{restore, sha256_file};
use super::error::PatchError;
use crate::app::{exit, AppDeps, CoreResult, Outcome};
use crate::cli::RestoreArgs;

#[derive(Debug, Serialize)]
struct RestoredFile {
    path: PathBuf,
    source: PathBuf,
    /// None: the file did not exist.
    sha256_before: Option<String>,
    sha256_source: String,
    sha256_after: String,
    /// false: already identical to the source, left untouched.
    rewritten: bool,
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

/// Shared by the CLI and the `surgeon_restore` MCP tool (which sandboxes the paths first).
pub fn core_restore(args: &RestoreArgs, deps: &AppDeps) -> CoreResult {
    if args.from_origin.is_some() && (!args.backup.is_empty() || args.backup_dir.is_some()) {
        return Err(invalid(
            "invalid_arguments",
            "from_origin exclut backup et backup_dir",
        ));
    }
    if !args.backup.is_empty() && args.backup_dir.is_some() {
        return Err(invalid("invalid_arguments", "backup exclut backup_dir"));
    }
    let (mode, pairs) = match &args.from_origin {
        Some(origin) => ("origin", origin_pairs(origin, &args.model)?),
        None if args.backup.is_empty() => (
            "backup",
            newest_backups(&args.model, args.backup_dir.as_deref())?,
        ),
        None => ("backup", explicit_backups(&args.backup, &args.model)?),
    };
    let mut files = Vec::with_capacity(pairs.len());
    for (source, target) in &pairs {
        let done = files.len();
        let failed = |message: String| {
            let message = format!(
                "{message} ({done} fichier(s) déjà restauré(s) sur {})",
                pairs.len()
            );
            Outcome::fail(exit::EXECUTION_FAILED, message).with_kind("execution_failed")
        };
        if deps.interrupted.load(Ordering::SeqCst) {
            return Err(failed("interrompu".into()).with_kind("cancelled"));
        }
        files.push(restore_one(source, target).map_err(|e| failed(e.to_string()))?);
    }
    let rewritten = files.iter().filter(|f| f.rewritten).count();
    let mut human = format!(
        "[RESTORE]  {} · {} fichier(s) · {rewritten} réécrit(s) · SHA-256 identique à la source ✓\n",
        args.model.display(),
        files.len()
    );
    for f in &files {
        human.push_str(&format!(
            "           {} <- {} · {}\n",
            f.path.display(),
            f.source.display(),
            &f.sha256_after[..12]
        ));
    }
    Ok(Outcome::ok(
        json!({ "mode": mode, "model": args.model, "files": files, "identical": true }),
        human,
    ))
}

fn restore_one(source: &Path, target: &Path) -> Result<RestoredFile, PatchError> {
    let sha256_source = sha256_file(source)?;
    let sha256_before = match target.is_file() {
        true => Some(sha256_file(target)?),
        false => None,
    };
    let rewritten = sha256_before.as_deref() != Some(sha256_source.as_str());
    let sha256_after = match rewritten {
        true => {
            if let Some(dir) = target.parent() {
                fs::create_dir_all(dir).map_err(|e| PatchError::io(dir, &e))?;
            }
            restore(source, target).map_err(|e| PatchError::io(target, &e))?;
            sha256_file(target)?
        }
        false => sha256_source.clone(),
    };
    if sha256_after != sha256_source {
        return Err(PatchError::Backup(format!(
            "SHA-256 après restauration différent de la source : {}",
            target.display()
        )));
    }
    Ok(RestoredFile {
        path: target.to_path_buf(),
        source: source.to_path_buf(),
        sha256_before,
        sha256_source,
        sha256_after,
        rewritten,
    })
}

/// (source, target) of each file to restore.
type Pairs = Vec<(PathBuf, PathBuf)>;
/// (unix secs, n) of `<name>.bak-<secs>[-n]`: the order of creation.
type Order = (u64, u32);

/// `<original>.bak-<secs>[-n]` -> (original, order).
fn parse_backup_name(name: &str) -> Option<(&str, Order)> {
    let at = name.rfind(".bak-")?;
    let suffix = &name[at + ".bak-".len()..];
    let (secs, n) = match suffix.split_once('-') {
        Some((secs, n)) => (secs, n.parse().ok()?),
        None => (suffix, 0),
    };
    let original = &name[..at];
    (!original.is_empty()).then_some((original, (secs.parse().ok()?, n)))
}

/// The file a backup puts back: the model itself, or its file of that name in a folder.
fn target_of(model: &Path, original: &str) -> Option<PathBuf> {
    match model.is_dir() {
        true => Some(model.join(original)),
        false => (model.file_name() == Some(OsStr::new(original))).then(|| model.to_path_buf()),
    }
}

fn explicit_backups(backups: &[PathBuf], model: &Path) -> Result<Pairs, Outcome> {
    backups
        .iter()
        .map(|backup| {
            let name = backup.file_name().and_then(OsStr::to_str).unwrap_or("");
            let (original, _) = parse_backup_name(name).ok_or_else(|| {
                invalid(
                    "invalid_arguments",
                    format!(
                        "{} : pas une sauvegarde de apply (<nom>.bak-<horodatage>)",
                        backup.display()
                    ),
                )
            })?;
            let target = target_of(model, original).ok_or_else(|| {
                invalid(
                    "invalid_arguments",
                    format!(
                        "{} sauvegarde {original}, pas {}",
                        backup.display(),
                        model.display()
                    ),
                )
            })?;
            Ok((backup.clone(), target))
        })
        .collect()
}

fn newest_backups(model: &Path, backup_dir: Option<&Path>) -> Result<Pairs, Outcome> {
    let dir = match (backup_dir, model.is_dir()) {
        (Some(dir), _) => dir.to_path_buf(),
        (None, true) => model.to_path_buf(),
        (None, false) => model.parent().map(Path::to_path_buf).unwrap_or_default(),
    };
    let entries = fs::read_dir(&dir)
        .map_err(|e| invalid("invalid_input", format!("{} : {e}", dir.display())))?;
    // Target -> its backups with their order of creation.
    let mut found: BTreeMap<PathBuf, Vec<(Order, PathBuf)>> = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some((original, order)) = path
            .file_name()
            .and_then(OsStr::to_str)
            .and_then(parse_backup_name)
        else {
            continue;
        };
        // In a folder, only the backups of its own files (a backup_dir may hold others).
        match target_of(model, original) {
            Some(target) if path.is_file() && (!model.is_dir() || target.is_file()) => {
                found.entry(target).or_default().push((order, path))
            }
            _ => {}
        }
    }
    if found.is_empty() {
        return Err(invalid(
            "backup_not_found",
            format!(
                "aucune sauvegarde <nom>.bak-* de {} dans {}",
                model.display(),
                dir.display()
            ),
        ));
    }
    if model.is_dir() && found.values().any(|backups| backups.len() > 1) {
        return Err(invalid(
            "ambiguous_backup",
            format!(
                "plusieurs apply sauvegardés dans {} : nommer les sauvegardes voulues (--backup, les data.backups de apply) ou repartir de l'origine (--from-origin)",
                dir.display()
            ),
        ));
    }
    Ok(found
        .into_iter()
        .filter_map(|(target, backups)| Some((backups.into_iter().max()?.1, target)))
        .collect())
}

fn origin_pairs(origin: &Path, model: &Path) -> Result<Pairs, Outcome> {
    let mismatch = || {
        invalid(
            "invalid_arguments",
            format!(
                "{} et {} : un fichier et un dossier",
                origin.display(),
                model.display()
            ),
        )
    };
    if origin.is_file() {
        return match model.is_dir() {
            true => Err(mismatch()),
            false => Ok(vec![(origin.to_path_buf(), model.to_path_buf())]),
        };
    }
    if model.is_file() {
        return Err(mismatch());
    }
    let mut pairs = Vec::new();
    walk(origin, model, &mut pairs)
        .map_err(|e| invalid("invalid_input", format!("{} : {e}", origin.display())))?;
    match pairs.is_empty() {
        true => Err(invalid(
            "invalid_input",
            format!("{} : aucun fichier", origin.display()),
        )),
        false => Ok(pairs),
    }
}

/// Every file under `from`, with the same relative path under `to`.
fn walk(from: &Path, to: &Path, pairs: &mut Pairs) -> io::Result<()> {
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        match entry.path().is_dir() {
            true => walk(&entry.path(), &target, pairs)?,
            false => pairs.push((entry.path(), target)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backup_names_parse_in_creation_order() {
        assert_eq!(
            parse_backup_name("m.safetensors.bak-17"),
            Some(("m.safetensors", (17, 0)))
        );
        assert_eq!(
            parse_backup_name("m.safetensors.bak-17-2"),
            Some(("m.safetensors", (17, 2)))
        );
        assert_eq!(parse_backup_name("m.safetensors"), None);
        assert_eq!(parse_backup_name(".bak-17"), None);
        assert_eq!(parse_backup_name("m.bak-x"), None);
        assert!(parse_backup_name("m.bak-9").unwrap().1 < parse_backup_name("m.bak-10").unwrap().1);
    }

    #[test]
    fn a_folder_with_two_applies_on_one_file_is_ambiguous() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.safetensors"), b"a").unwrap();
        fs::write(dir.path().join("a.safetensors.bak-1"), b"0").unwrap();
        let single = newest_backups(dir.path(), None).unwrap();
        assert_eq!(
            single,
            vec![(
                dir.path().join("a.safetensors.bak-1"),
                dir.path().join("a.safetensors")
            )]
        );
        fs::write(dir.path().join("a.safetensors.bak-2"), b"1").unwrap();
        let err = newest_backups(dir.path(), None).unwrap_err();
        assert_eq!(err.error.unwrap().kind.as_deref(), Some("ambiguous_backup"));
        // A single file takes its newest backup.
        let file = newest_backups(&dir.path().join("a.safetensors"), None).unwrap();
        assert_eq!(file[0].0, dir.path().join("a.safetensors.bak-2"));
    }
}
