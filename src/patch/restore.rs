//! `surgeon restore` / `surgeon_restore`: puts a model back byte for byte, from the backups
//! of `apply` or from the origin model. Every file is hashed before and after; the restore
//! fails unless the final SHA-256 is the source's. Sources are only read, never removed.
//!
//! Detection without `--backup`: every backup of one `apply` carries the same checkpoint
//! `<name>.bak-<secs>-<n>` (see `backup`), so the newest checkpoint is exactly the state before
//! the last `apply`, however many came before. Folders written by older versions have one
//! stamp per file: with a single backed up file its newest backup is that same state; with
//! several, nothing links them, and each file's oldest backup is taken instead, which is a
//! consistent state too: the one before the first `apply` (`data.checkpoint` says which).

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use serde::Serialize;
use serde_json::json;

use super::backup::{backups_in, parse_backup_name, restore, sha256_file, Checkpoint};
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
    let (mode, pairs, checkpoint) = match &args.from_origin {
        Some(origin) => ("origin", origin_pairs(origin, &args.model)?, None),
        None if args.backup.is_empty() => {
            let (pairs, checkpoint) = detected_backups(&args.model, args.backup_dir.as_deref())?;
            ("backup", pairs, Some(checkpoint))
        }
        None => ("backup", explicit_backups(&args.backup, &args.model)?, None),
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
    let mut data = json!({ "mode": mode, "model": args.model, "files": files, "identical": true });
    let mut warnings = Vec::new();
    if let Some(checkpoint) = checkpoint {
        data["checkpoint"] = json!(checkpoint);
        if checkpoint == FIRST_APPLY {
            warnings.push(
                "sauvegardes d'une ancienne version : retour à l'état d'avant le premier apply"
                    .to_owned(),
            );
        }
    }
    let mut outcome = Outcome::ok(data, human);
    outcome.warnings = warnings;
    Ok(outcome)
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
/// What the detection went back to (`data.checkpoint`).
const LAST_APPLY: &str = "last_apply";
const FIRST_APPLY: &str = "first_apply";

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

/// The newest checkpoint of the model, see the module documentation.
fn detected_backups(
    model: &Path,
    backup_dir: Option<&Path>,
) -> Result<(Pairs, &'static str), Outcome> {
    let dir = match (backup_dir, model.is_dir()) {
        (Some(dir), _) => dir.to_path_buf(),
        (None, true) => model.to_path_buf(),
        (None, false) => model.parent().map(Path::to_path_buf).unwrap_or_default(),
    };
    let backups = backups_in(&dir)
        .map_err(|e| invalid("invalid_input", format!("{} : {e}", dir.display())))?;
    // Target -> its backups with their checkpoints.
    let mut found: BTreeMap<PathBuf, Vec<(Checkpoint, PathBuf)>> = BTreeMap::new();
    for backup in backups {
        // In a folder, only the backups of its own files (a backup_dir may hold others).
        match target_of(model, &backup.original) {
            Some(target) if !model.is_dir() || target.is_file() => found
                .entry(target)
                .or_default()
                .push((backup.checkpoint, backup.path)),
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
    let newest = found
        .values()
        .flatten()
        .map(|(checkpoint, _)| *checkpoint)
        .max()
        .unwrap_or_default();
    if newest.1 > 0 || found.len() == 1 {
        let pairs = found
            .into_iter()
            .filter_map(|(target, backups)| {
                let (_, path) = backups.into_iter().find(|(c, _)| *c == newest)?;
                Some((path, target))
            })
            .collect();
        return Ok((pairs, LAST_APPLY));
    }
    let pairs = found
        .into_iter()
        .filter_map(|(target, backups)| Some((backups.into_iter().min()?.1, target)))
        .collect();
    Ok((pairs, FIRST_APPLY))
}

/// A file against a folder means that file of the folder: the weights picked as a project's
/// source, restored over its working copy.
fn origin_pairs(origin: &Path, model: &Path) -> Result<Pairs, Outcome> {
    if origin.is_file() {
        let target = match (model.is_dir(), origin.file_name()) {
            (true, Some(name)) => model.join(name),
            _ => model.to_path_buf(),
        };
        return Ok(vec![(origin.to_path_buf(), target)]);
    }
    if model.is_file() {
        let source = model.file_name().map(|name| origin.join(name));
        return match source.filter(|s| s.is_file()) {
            Some(source) => Ok(vec![(source, model.to_path_buf())]),
            None => Err(invalid(
                "invalid_input",
                format!("{} : pas de fichier {}", origin.display(), model.display()),
            )),
        };
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

/// Every visible file under `from`, with the same relative path under `to`. Hidden entries
/// (`.cache/huggingface`, `.git`, `.gitattributes`) are left out, as in a working copy.
fn walk(from: &Path, to: &Path, pairs: &mut Pairs) -> io::Result<()> {
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
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

    fn touch(dir: &Path, names: &[&str]) {
        for name in names {
            fs::write(dir.join(name), name.as_bytes()).unwrap();
        }
    }

    #[test]
    fn the_newest_checkpoint_is_the_state_before_the_last_apply() {
        let dir = tempfile::tempdir().unwrap();
        // Apply 1 touched a and b, apply 2 only a: b must not go back to apply 1.
        touch(
            dir.path(),
            &["a", "b", "a.bak-5-1", "b.bak-5-1", "a.bak-5-2"],
        );
        let (pairs, checkpoint) = detected_backups(dir.path(), None).unwrap();
        assert_eq!(checkpoint, LAST_APPLY);
        assert_eq!(
            pairs,
            vec![(dir.path().join("a.bak-5-2"), dir.path().join("a"))]
        );
    }

    #[test]
    fn older_folders_stay_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        // One backed up file: its newest backup, however many there are.
        touch(dir.path(), &["a", "a.bak-1", "a.bak-2"]);
        let (pairs, checkpoint) = detected_backups(dir.path(), None).unwrap();
        assert_eq!(pairs[0].0, dir.path().join("a.bak-2"));
        assert_eq!(checkpoint, LAST_APPLY);
        let (file, _) = detected_backups(&dir.path().join("a"), None).unwrap();
        assert_eq!(file[0].0, dir.path().join("a.bak-2"));
        // Several files without a shared checkpoint: all back to before the first apply.
        touch(dir.path(), &["b", "b.bak-3"]);
        let (pairs, checkpoint) = detected_backups(dir.path(), None).unwrap();
        assert_eq!(checkpoint, FIRST_APPLY);
        assert_eq!(
            pairs,
            vec![
                (dir.path().join("a.bak-1"), dir.path().join("a")),
                (dir.path().join("b.bak-3"), dir.path().join("b")),
            ]
        );
    }

    #[test]
    fn an_origin_file_goes_over_its_name_in_the_copy() {
        let root = tempfile::tempdir().unwrap();
        let (origin, copy) = (root.path().join("origin"), root.path().join("copy"));
        fs::create_dir_all(&origin).unwrap();
        fs::create_dir_all(&copy).unwrap();
        touch(&origin, &["m.safetensors"]);
        touch(&copy, &["m.safetensors"]);
        let weights = origin.join("m.safetensors");
        assert_eq!(
            origin_pairs(&weights, &copy).unwrap(),
            vec![(weights.clone(), copy.join("m.safetensors"))]
        );
        assert_eq!(
            origin_pairs(&origin, &copy.join("m.safetensors")).unwrap(),
            vec![(weights, copy.join("m.safetensors"))]
        );
    }

    #[test]
    fn hidden_entries_of_the_origin_are_not_restored() {
        let root = tempfile::tempdir().unwrap();
        let (origin, copy) = (root.path().join("origin"), root.path().join("copy"));
        let cache = origin.join(".cache").join("huggingface");
        fs::create_dir_all(&cache).unwrap();
        fs::create_dir_all(origin.join("sub")).unwrap();
        touch(&origin, &["m.safetensors", ".gitattributes"]);
        touch(&cache, &["download.lock"]);
        touch(&origin.join("sub"), &["extra.json"]);
        let mut pairs = origin_pairs(&origin, &copy).unwrap();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                (origin.join("m.safetensors"), copy.join("m.safetensors")),
                (
                    origin.join("sub").join("extra.json"),
                    copy.join("sub").join("extra.json")
                ),
            ]
        );
    }
}
