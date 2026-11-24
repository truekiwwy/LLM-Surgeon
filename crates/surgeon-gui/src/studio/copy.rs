//! Working copies: the only models the studio writes. `copies/<id>/` is filled as
//! `copies/<id>.partial/` and renamed when complete, so a failed or cancelled copy leaves no
//! file behind. `std::fs::copy` asks the OS for a clone where the file system has one
//! (clonefile on APFS, copy_file_range on btrfs / XFS, CopyFileEx on ReFS / Dev Drive), and
//! copies the bytes elsewhere.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Kept free after the copy: the apply backups and the run files need room too.
pub const MARGIN_BYTES: u64 = 256 << 20;

#[derive(Debug, Clone, PartialEq)]
pub struct CopyPlan {
    /// Source files, in name order: the file itself, or a directory's top-level files.
    pub files: Vec<PathBuf>,
    pub bytes: u64,
}

/// What a copy of `source` holds (a model directory is flat: config, tokenizer, shards).
pub fn plan(source: &Path) -> Result<CopyPlan, String> {
    let io = |e: std::io::Error| format!("{} : {e}", source.display());
    let files: Vec<PathBuf> = match source.is_dir() {
        true => {
            let mut files: Vec<PathBuf> = std::fs::read_dir(source)
                .map_err(io)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect();
            files.sort();
            files
        }
        false => vec![source.to_path_buf()],
    };
    sized(files)
}

fn sized(files: Vec<PathBuf>) -> Result<CopyPlan, String> {
    let mut bytes = 0;
    for f in &files {
        bytes += std::fs::metadata(f)
            .map_err(|e| format!("{} : {e}", f.display()))?
            .len();
    }
    Ok(CopyPlan { files, bytes })
}

/// The folder a model lives in: `source` itself, or the parent of a weights file picked inside
/// a Hugging Face model folder (one holding `config.json`), whose companions inference needs.
pub fn model_root(source: &Path) -> &Path {
    match source.parent() {
        Some(dir) if source.is_file() && dir.join("config.json").is_file() => dir,
        _ => source,
    }
}

/// What a working copy holds: every visible top-level file of [`model_root`] (config,
/// tokenizer, chat template, shard index, shards), never dotfiles like `.gitattributes`.
pub fn copy_plan(source: &Path) -> Result<CopyPlan, String> {
    let mut files = plan(model_root(source))?.files;
    files.retain(|f| {
        !f.file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
    });
    sized(files)
}

/// Free bytes on the volume holding `dir` (the mount point that is its longest prefix).
pub fn free_space(dir: &Path) -> Option<u64> {
    // Not canonicalize: its verbatim `\\?\` prefix on Windows never matches a mount point.
    let dir = std::path::absolute(dir).ok()?;
    let disks = sysinfo::Disks::new_with_refreshed_list().ok()?;
    disks
        .iter()
        .filter(|d| dir.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

/// Refuses a copy that would leave less than [`MARGIN_BYTES`] free.
pub fn check_space(needed: u64, free: u64) -> Result<(), String> {
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    match needed.saturating_add(MARGIN_BYTES) <= free {
        true => Ok(()),
        false => Err(format!(
            "espace disque insuffisant : copie de {:.2} Gio + marge {:.2} Gio, {:.2} Gio libres",
            gib(needed),
            gib(MARGIN_BYTES),
            gib(free)
        )),
    }
}

/// Copies `source` to `<copies>/<id>/` once `free` bytes are known to be enough, calling
/// `progress(done_files, total_files)` after each file. Returns the model path to use: the
/// copied directory, or the copied file inside it.
pub fn make(
    source: &Path,
    copies: &Path,
    id: &str,
    free: u64,
    cancel: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
) -> Result<PathBuf, String> {
    let plan = copy_plan(source)?;
    check_space(plan.bytes, free)?;
    let target = copies.join(id);
    if target.exists() {
        return Err(format!("{} existe déjà", target.display()));
    }
    let partial = copies.join(format!("{id}.partial"));
    // A leftover of a killed studio: never part of a finished copy.
    let _ = std::fs::remove_dir_all(&partial);
    let result = fill(&plan, &partial, cancel, &mut progress)
        .and_then(|()| std::fs::rename(&partial, &target).map_err(|e| e.to_string()));
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&partial);
        return Err(e);
    }
    Ok(match source.is_dir() {
        true => target,
        false => target.join(source.file_name().unwrap_or_default()),
    })
}

fn fill(
    plan: &CopyPlan,
    partial: &Path,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(usize, usize),
) -> Result<(), String> {
    std::fs::create_dir_all(partial).map_err(|e| format!("{} : {e}", partial.display()))?;
    for (i, file) in plan.files.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err("copie annulée".into());
        }
        let name = file.file_name().unwrap_or_default();
        std::fs::copy(file, partial.join(name)).map_err(|e| format!("{} : {e}", file.display()))?;
        progress(i + 1, plan.files.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use weight_surgeon::pipeline::job::sha256_dir;

    fn model(dir: &Path) -> PathBuf {
        let m = dir.join("model");
        std::fs::create_dir_all(&m).unwrap();
        std::fs::write(m.join("config.json"), b"{}").unwrap();
        std::fs::write(m.join("model.safetensors"), vec![7u8; 4096]).unwrap();
        m
    }

    #[test]
    fn a_copy_is_complete_and_the_original_is_unchanged() {
        let ws = tempfile::tempdir().unwrap();
        let src = model(ws.path());
        let before = sha256_dir(&src).unwrap();
        let copies = ws.path().join("copies");
        std::fs::create_dir_all(&copies).unwrap();
        let mut calls = Vec::new();
        let never = AtomicBool::new(false);
        let copy = make(&src, &copies, "c1", u64::MAX, &never, |d, t| {
            calls.push((d, t))
        })
        .unwrap();
        assert_eq!(copy, copies.join("c1"));
        assert_eq!(sha256_dir(&copy).unwrap(), before);
        // Writing the copy leaves the original alone.
        std::fs::write(copy.join("model.safetensors"), b"patched").unwrap();
        assert_eq!(sha256_dir(&src).unwrap(), before);
        assert_eq!(calls, vec![(1, 2), (2, 2)]);
        assert!(
            make(&src, &copies, "c1", u64::MAX, &never, |_, _| {}).is_err(),
            "exists"
        );
    }

    #[test]
    fn the_hugging_face_cache_of_the_origin_is_not_copied() {
        let ws = tempfile::tempdir().unwrap();
        let src = model(ws.path());
        let cache = src.join(".cache").join("huggingface");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(cache.join("model.safetensors.metadata"), vec![1u8; 4096]).unwrap();
        let copies = ws.path().join("copies");
        std::fs::create_dir_all(&copies).unwrap();
        let never = AtomicBool::new(false);
        let copy = make(&src, &copies, "copy-1", u64::MAX, &never, |_, _| {}).unwrap();
        assert!(!copy.join(".cache").exists());
        assert!(copy.join("model.safetensors").is_file());
        assert_eq!(copy_plan(&src).unwrap().bytes, 4096 + 2);
    }

    #[test]
    fn a_single_file_model_is_copied_into_its_own_folder() {
        let ws = tempfile::tempdir().unwrap();
        let src = ws.path().join("m.gguf");
        std::fs::write(&src, b"gguf").unwrap();
        let never = AtomicBool::new(false);
        let copy = make(&src, ws.path(), "c", u64::MAX, &never, |_, _| {}).unwrap();
        assert_eq!(copy, ws.path().join("c").join("m.gguf"));
        assert_eq!(std::fs::read(copy).unwrap(), b"gguf");
    }

    #[test]
    fn a_weights_file_picked_in_a_model_folder_brings_its_companions() {
        let ws = tempfile::tempdir().unwrap();
        let src = model(ws.path());
        for name in ["tokenizer.json", "chat_template.jinja", ".gitattributes"] {
            std::fs::write(src.join(name), name).unwrap();
        }
        std::fs::create_dir_all(src.join(".cache")).unwrap();
        let weights = src.join("model.safetensors");
        // The studio holds the source open for reading while it copies.
        #[cfg(windows)]
        let _held = super::super::lock::lock(&weights).unwrap();
        let never = AtomicBool::new(false);
        let copy = make(&weights, ws.path(), "c", u64::MAX, &never, |_, _| {}).unwrap();
        assert_eq!(copy, ws.path().join("c").join("model.safetensors"));
        let mut names: Vec<_> = std::fs::read_dir(ws.path().join("c"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "chat_template.jinja",
                "config.json",
                "model.safetensors",
                "tokenizer.json"
            ]
        );
    }

    #[test]
    fn too_little_space_is_refused_before_anything_is_written() {
        let ws = tempfile::tempdir().unwrap();
        let src = model(ws.path());
        let copies = ws.path().join("copies");
        std::fs::create_dir_all(&copies).unwrap();
        let free = 4096 + MARGIN_BYTES; // the files need 4098 bytes
        let never = AtomicBool::new(false);
        let e = make(&src, &copies, "c", free, &never, |_, _| panic!("copied")).unwrap_err();
        assert!(e.contains("insuffisant"), "{e}");
        assert_eq!(std::fs::read_dir(&copies).unwrap().count(), 0);
        check_space(4098, 4098 + MARGIN_BYTES).unwrap();
    }

    #[test]
    fn a_cancelled_copy_leaves_no_partial_file() {
        let ws = tempfile::tempdir().unwrap();
        let src = model(ws.path());
        let copies = ws.path().join("copies");
        std::fs::create_dir_all(&copies).unwrap();
        let cancel = AtomicBool::new(false);
        let e = make(&src, &copies, "c", u64::MAX, &cancel, |_, _| {
            cancel.store(true, Ordering::SeqCst)
        })
        .unwrap_err();
        assert!(e.contains("annulée"));
        assert_eq!(std::fs::read_dir(&copies).unwrap().count(), 0);
    }

    #[test]
    fn the_free_space_of_an_existing_folder_is_known() {
        let ws = tempfile::tempdir().unwrap();
        assert!(free_space(ws.path()).is_some_and(|b| b > 0));
    }
}
