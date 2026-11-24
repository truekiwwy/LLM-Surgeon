use std::collections::BTreeMap;
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use super::sweep::{self, Candidate, Decision, Lab, Score};
use super::*;
use crate::pipeline::job::sha256_dir;

fn model(root: &Path) -> PathBuf {
    let dir = root.join("copies").join("c1");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), b"{}").unwrap();
    fs::write(dir.join("model.safetensors"), b"base-weights").unwrap();
    dir
}

fn store(root: &Path) -> Store {
    Store::in_work(&root.join("work"))
}

fn write_in_place(path: &Path, bytes: &[u8]) {
    let mut file = fs::OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(bytes).unwrap();
    file.set_len(bytes.len() as u64).unwrap();
}

fn identity(path: &Path) -> crate::kernel::FileIdentity {
    file_identity(path).unwrap()
}

#[test]
fn create_hard_links_every_file_under_a_sha256_manifest() {
    let root = tempfile::tempdir().unwrap();
    let (model, store) = (model(root.path()), store(root.path()));
    fs::write(model.join("model.safetensors.bak-1-0"), b"old").unwrap();

    let m = store
        .create(&model, " first ", BTreeMap::from([("gain".into(), 0.5)]))
        .unwrap();

    assert_eq!(
        (m.id.as_str(), m.label.as_str(), m.parent.as_deref()),
        ("0001", "first", None)
    );
    fs::remove_file(model.join("model.safetensors.bak-1-0")).unwrap();
    assert_eq!(m.sha256, sha256_dir(&model).unwrap());
    let names: Vec<_> = m.files.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        ["config.json", "model.safetensors"],
        "backups left out"
    );
    for f in &m.files {
        let (copy, snap) = (model.join(&f.name), store.model_dir(&m.id).join(&f.name));
        assert!(f.linked);
        assert_eq!(identity(&copy).key(), identity(&snap).key(), "{}", f.name);
        assert_eq!(identity(&snap).links, 2);
        assert_eq!(f.sha256, sha256_file(&snap).unwrap());
    }
    assert!(store.dir().join(MARKER).is_file());
    assert_eq!(store.head(&model).unwrap().as_deref(), Some("0001"));
    assert_eq!(store.verify("0001").unwrap(), m);
}

#[test]
fn an_in_place_write_after_unshare_leaves_the_checkpoint_intact() {
    let root = tempfile::tempdir().unwrap();
    let (model, store) = (model(root.path()), store(root.path()));
    let m = store.create(&model, "base", BTreeMap::new()).unwrap();
    let weights = model.join("model.safetensors");

    assert!(unshare(&weights).unwrap());
    assert!(!unshare(&weights).unwrap(), "already private");
    assert_eq!(identity(&weights).links, 1);
    write_in_place(&weights, b"patched");

    assert_eq!(fs::read(&weights).unwrap(), b"patched");
    assert_eq!(
        fs::read(store.model_dir(&m.id).join("model.safetensors")).unwrap(),
        b"base-weights"
    );
    store.verify(&m.id).unwrap();

    let err = store.check_model(&m.id, &model).unwrap_err();
    assert_eq!(err.kind(), "checksum_mismatch");
    fs::write(model.join("model.safetensors.bak-1-0"), b"backup").unwrap();
    store.restore(&m.id, &model).unwrap();
    store.check_model(&m.id, &model).unwrap();
}

#[test]
fn restore_relinks_the_copy_and_keeps_its_backups_and_folders() {
    let root = tempfile::tempdir().unwrap();
    let (model, store) = (model(root.path()), store(root.path()));
    let base = store.create(&model, "base", BTreeMap::new()).unwrap();
    let weights = model.join("model.safetensors");
    unshare(&weights).unwrap();
    write_in_place(&weights, b"patched");
    fs::write(model.join("extra.json"), b"new").unwrap();
    fs::write(model.join("model.safetensors.bak-9-0"), b"backup").unwrap();
    fs::create_dir_all(model.join("notes")).unwrap();
    fs::write(model.join("notes").join("a.txt"), b"kept").unwrap();
    let patched = store.create(&model, "patched", BTreeMap::new()).unwrap();
    assert_eq!(patched.parent.as_deref(), Some("0001"));

    let restored = store.restore(&base.id, &model).unwrap();

    assert_eq!(restored.sha256, base.sha256);
    assert!(restored.linked);
    assert_eq!(fs::read(&weights).unwrap(), b"base-weights");
    assert!(!model.join("extra.json").exists(), "not in the checkpoint");
    assert_eq!(
        fs::read(model.join("model.safetensors.bak-9-0")).unwrap(),
        b"backup"
    );
    assert_eq!(
        fs::read(model.join("notes").join("a.txt")).unwrap(),
        b"kept"
    );
    let snap = store.model_dir(&base.id).join("model.safetensors");
    assert_eq!(identity(&weights).key(), identity(&snap).key());
    assert_eq!(store.head(&model).unwrap().as_deref(), Some("0001"));
    store.verify(&patched.id).unwrap();
    let parent = model.parent().unwrap();
    let leftovers: Vec<_> = fs::read_dir(parent)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(
        leftovers.len(),
        1,
        "no staging nor retired folder: {leftovers:?}"
    );

    let third = store.create(&model, "again", BTreeMap::new()).unwrap();
    assert_eq!(
        third.parent.as_deref(),
        Some("0001"),
        "restore moved the head"
    );
}

#[test]
fn list_gives_the_tree_and_the_real_disk_footprint() {
    let root = tempfile::tempdir().unwrap();
    let (model, store) = (model(root.path()), store(root.path()));
    let a = store.create(&model, "a", BTreeMap::new()).unwrap();
    let b = store.create(&model, "b", BTreeMap::new()).unwrap();
    let weights = model.join("model.safetensors");
    unshare(&weights).unwrap();
    write_in_place(&weights, b"patched-weights!");
    let c = store.create(&model, "c", BTreeMap::new()).unwrap();

    let listing = store.list().unwrap();

    let ids: Vec<_> = listing
        .checkpoints
        .iter()
        .map(|l| l.manifest.id.as_str())
        .collect();
    assert_eq!(ids, ["0001", "0002", "0003"]);
    assert_eq!(listing.checkpoints[0].children, ["0002"]);
    assert_eq!(listing.checkpoints[1].children, ["0003"]);
    assert_eq!(c.parent.as_deref(), Some(b.id.as_str()));
    assert_eq!(listing.logical_bytes, a.bytes() + b.bytes() + c.bytes());
    assert_eq!(
        listing.disk_bytes,
        2 + 12 + 16,
        "config.json and the base weights once"
    );
    let exclusive: Vec<_> = listing
        .checkpoints
        .iter()
        .map(|l| l.exclusive_bytes)
        .collect();
    assert_eq!(exclusive, [0, 0, 16]);
}

#[test]
fn a_damaged_or_foreign_checkpoint_is_refused_and_the_copy_untouched() {
    let root = tempfile::tempdir().unwrap();
    let (model, store) = (model(root.path()), store(root.path()));
    let m = store.create(&model, "base", BTreeMap::new()).unwrap();
    let other = root.path().join("copies").join("c2");
    fs::create_dir_all(&other).unwrap();
    fs::write(other.join("model.safetensors"), b"x").unwrap();

    let foreign = store.restore(&m.id, &other).unwrap_err();
    assert_eq!(foreign.kind(), "invalid_input");
    assert_eq!(
        store.restore("../x", &model).unwrap_err().kind(),
        "invalid_input"
    );
    assert_eq!(
        store.restore("0042", &model).unwrap_err().kind(),
        "invalid_input"
    );
    let inside = store.model_dir(&m.id);
    assert_eq!(
        store
            .create(&inside, "x", BTreeMap::new())
            .unwrap_err()
            .kind(),
        "invalid_input"
    );

    let weights = model.join("model.safetensors");
    unshare(&weights).unwrap();
    fs::write(
        store.model_dir(&m.id).join("model.safetensors"),
        b"tampered",
    )
    .unwrap();
    write_in_place(&weights, b"patched");
    let damaged = store.restore(&m.id, &model).unwrap_err();
    assert_eq!(damaged.kind(), "checksum_mismatch");
    assert_eq!(fs::read(&weights).unwrap(), b"patched");
}

#[test]
fn a_single_weights_file_is_restored_through_one_rename() {
    let root = tempfile::tempdir().unwrap();
    let store = store(root.path());
    let file = root.path().join("copies").join("m.gguf");
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(&file, b"gguf-base").unwrap();
    let m = store.create(&file, "base", BTreeMap::new()).unwrap();
    assert!(m.single_file);
    unshare(&file).unwrap();
    write_in_place(&file, b"gguf-new");

    store.restore(&m.id, &file).unwrap();

    assert_eq!(fs::read(&file).unwrap(), b"gguf-base");
    assert_eq!(
        identity(&file).key(),
        identity(&store.model_dir(&m.id).join("m.gguf")).key()
    );
}

struct FakeLab {
    weights: PathBuf,
    applied: Vec<f32>,
    fail_at: Option<f32>,
}

fn outcome(alpha: f32) -> (f64, f64) {
    match alpha {
        0.5 => (0.70, 0.10),
        0.8 => (0.80, 0.25),
        1.0 => (0.80, 0.28),
        1.2 => (0.95, 0.60),
        _ => (0.50, 0.00),
    }
}

impl Lab for FakeLab {
    fn check(&mut self, _: &Path, c: &Candidate) -> Result<()> {
        match c.alpha > 0.0 {
            true => Ok(()),
            false => Err(CheckpointError::Invalid("alpha".into())),
        }
    }

    fn apply(&mut self, _: &Path, c: &Candidate) -> Result<()> {
        assert_eq!(
            fs::read(&self.weights).unwrap(),
            b"base-weights",
            "rolled back first"
        );
        unshare(&self.weights).unwrap();
        write_in_place(&self.weights, c.alpha.to_string().as_bytes());
        self.applied.push(c.alpha);
        match self.fail_at == Some(c.alpha) {
            true => Err(CheckpointError::Failed("apply".into())),
            false => Ok(()),
        }
    }

    fn measure(&mut self, _: &Path, base: bool) -> Result<Score> {
        let text = fs::read_to_string(&self.weights).unwrap();
        let alpha = match base {
            true => 0.0,
            false => text.parse().unwrap(),
        };
        let (target, drift) = outcome(alpha);
        Ok(Score { target, drift })
    }
}

fn candidates() -> Vec<Candidate> {
    sweep::grid(
        &[PathBuf::from("refusal.safetensors")],
        &["10-14".to_owned()],
        &[0.5, 0.8, 1.0, 1.2],
    )
}

#[test]
fn the_sweep_keeps_the_front_rolls_back_the_rest_and_ends_on_the_best() {
    let root = tempfile::tempdir().unwrap();
    let (model, store) = (model(root.path()), store(root.path()));
    let weights = model.join("model.safetensors");
    let mut lab = FakeLab {
        weights: weights.clone(),
        applied: Vec::new(),
        fail_at: None,
    };

    let report = sweep::run(
        &store,
        &model,
        &candidates(),
        0.3,
        &mut lab,
        &AtomicBool::new(false),
    )
    .unwrap();

    assert_eq!(lab.applied, [0.5, 0.8, 1.0, 1.2]);
    let decisions: Vec<_> = report.points.iter().map(|p| p.decision).collect();
    assert_eq!(
        decisions,
        [
            Decision::Kept,
            Decision::Kept,
            Decision::Dominated,
            Decision::DriftExceeded
        ]
    );
    assert_eq!(report.front, [0, 1]);
    assert_eq!(report.best, Some(1));
    assert!((report.points[1].gain - 0.30).abs() < 1e-12);
    assert_eq!(report.base_checkpoint, "0001");
    let kept: Vec<_> = report
        .points
        .iter()
        .map(|p| p.checkpoint.as_deref())
        .collect();
    assert_eq!(kept, [Some("0002"), Some("0003"), None, None]);
    assert_eq!(report.restored, "0003");
    assert_eq!(fs::read(&weights).unwrap(), b"0.8");
    let listing = store.list().unwrap();
    assert_eq!(
        listing.checkpoints.len(),
        3,
        "no checkpoint for rejected points"
    );
    for l in &listing.checkpoints {
        store.verify(&l.manifest.id).unwrap();
        assert_eq!(l.manifest.parent.is_some(), l.manifest.id != "0001");
    }
    assert_eq!(listing.checkpoints[2].manifest.metrics["drift"], 0.25);
    assert_eq!(listing.checkpoints[0].manifest.metrics["target"], 0.50);
}

#[test]
fn a_failed_point_rolls_the_copy_back_and_nothing_starts_on_an_invalid_grid() {
    let root = tempfile::tempdir().unwrap();
    let (model, store) = (model(root.path()), store(root.path()));
    let weights = model.join("model.safetensors");
    let mut lab = FakeLab {
        weights: weights.clone(),
        applied: Vec::new(),
        fail_at: Some(0.8),
    };
    let never = AtomicBool::new(false);

    let failed = sweep::run(&store, &model, &candidates(), 0.3, &mut lab, &never).unwrap_err();
    assert_eq!(failed.kind(), "execution_failed");
    assert_eq!(fs::read(&weights).unwrap(), b"base-weights");

    let mut invalid = candidates();
    invalid[2].alpha = 0.0;
    lab.applied.clear();
    let refused = sweep::run(&store, &model, &invalid, 0.3, &mut lab, &never).unwrap_err();
    assert_eq!(refused.kind(), "invalid_input");
    assert!(lab.applied.is_empty(), "checked before any write");
    assert!(sweep::run(&store, &model, &candidates(), f64::NAN, &mut lab, &never).is_err());

    let cancelled = sweep::run(
        &store,
        &model,
        &candidates(),
        0.3,
        &mut lab,
        &AtomicBool::new(true),
    );
    assert_eq!(cancelled.unwrap_err().kind(), "cancelled");
    assert_eq!(fs::read(&weights).unwrap(), b"base-weights");
}
