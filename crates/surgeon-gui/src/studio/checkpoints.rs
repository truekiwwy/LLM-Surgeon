use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::json;
use weight_surgeon::checkpoint::pareto::{self, Point};
use weight_surgeon::checkpoint::{Listing, Manifest, Store};

use super::jobs::{Budgets, JobSpec};
use super::ledger::{Entry, Verdict};
use crate::agent::policy::canonical;

pub const TOOL: &str = "surgeon_checkpoint";
const FOLLOWED: [&str; 2] = ["surgeon_graph", "surgeon_bench"];

pub fn store(project: &Path) -> Store {
    Store::in_work(&project.join("work"))
}

pub fn stamp(project: &Path) -> Option<SystemTime> {
    let dir = store(project).dir().to_path_buf();
    std::fs::metadata(dir.join("heads.json"))
        .or_else(|_| std::fs::metadata(&dir))
        .and_then(|m| m.modified())
        .ok()
}

pub fn load(project: &Path) -> Result<Listing, String> {
    store(project).list().map_err(|e| e.to_string())
}

pub fn point(m: &Manifest) -> Option<Point> {
    Some(Point {
        gain: *m.metrics.get("gain")?,
        drift: *m.metrics.get("drift")?,
    })
}

pub fn front(listing: &Listing) -> Vec<String> {
    let measured: Vec<(&str, Point)> = listing
        .checkpoints
        .iter()
        .filter_map(|c| Some((c.manifest.id.as_str(), point(&c.manifest)?)))
        .collect();
    let points: Vec<Point> = measured.iter().map(|(_, p)| *p).collect();
    pareto::front(&points)
        .into_iter()
        .map(|i| measured[i].0.to_owned())
        .collect()
}

pub fn depths(listing: &Listing) -> Vec<usize> {
    let parent = |id: &str| {
        listing
            .checkpoints
            .iter()
            .find(|c| c.manifest.id == id)
            .and_then(|c| c.manifest.parent.as_deref())
    };
    listing
        .checkpoints
        .iter()
        .map(|c| {
            let mut depth = 0;
            let mut at = c.manifest.parent.as_deref();
            while let Some(id) = at.filter(|_| depth < listing.checkpoints.len()) {
                depth += 1;
                at = parent(id);
            }
            depth
        })
        .collect()
}

pub fn head<'a>(listing: &'a Listing, model: &Path) -> Option<&'a str> {
    let key = canonical(model);
    listing
        .heads
        .iter()
        .find(|(path, _)| canonical(Path::new(path)) == key)
        .map(|(_, id)| id.as_str())
}

pub fn restore_job(m: &Manifest, label: &str) -> JobSpec {
    JobSpec {
        tool: TOOL.into(),
        label: label.into(),
        args: json!({"action": "restore", "model": m.model, "id": m.id}),
        priority: 0,
        budgets: Budgets::default(),
        resumed_from: None,
        read_only: false,
    }
}

fn model_of(project: &Path, entry: &Entry) -> Option<PathBuf> {
    let model = Path::new(entry.params["model"].as_str()?);
    Some(canonical(&match model.is_absolute() {
        true => model.to_path_buf(),
        false => project.join(model),
    }))
}

pub fn followups(project: &Path, entries: &[Entry], restore: &Entry) -> Vec<JobSpec> {
    let Some(copy) = model_of(project, restore) else {
        return Vec::new();
    };
    FOLLOWED
        .iter()
        .filter_map(|tool| {
            entries
                .iter()
                .filter(|e| e.tool == *tool && e.verdict == Verdict::Ok)
                .filter(|e| e.finished_unix <= restore.started_unix)
                .filter(|e| model_of(project, e).as_ref() == Some(&copy))
                .max_by_key(|e| (e.finished_unix, e.run_id.clone()))
        })
        .map(|e| JobSpec {
            tool: e.tool.clone(),
            label: e.label.clone(),
            args: e.params.clone(),
            priority: 0,
            budgets: Budgets::default(),
            resumed_from: None,
            read_only: false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::{json, Value};

    use super::*;

    fn entry(id: &str, tool: &str, model: &str, finished: u64, verdict: Verdict) -> Entry {
        Entry {
            run_id: id.into(),
            tool: tool.into(),
            label: id.into(),
            params: json!({"model": model, "mode": "refusal"}),
            resumed_from: None,
            started_unix: finished.saturating_sub(5),
            finished_unix: finished,
            hashes: Default::default(),
            versions: Default::default(),
            verdict,
            exit_code: 0,
            result: Value::Null,
            error: None,
        }
    }

    #[test]
    fn a_restore_replays_the_newest_graph_and_bench_of_that_copy() {
        let project = tempfile::tempdir().unwrap();
        let copy = project.path().join("copies").join("c1");
        std::fs::create_dir_all(&copy).unwrap();
        let entries = vec![
            entry("g1", "surgeon_graph", "copies/c1", 10, Verdict::Ok),
            entry("g2", "surgeon_graph", "copies/c1", 20, Verdict::Ok),
            entry("g3", "surgeon_graph", "copies/c1", 30, Verdict::Failed),
            entry("b1", "surgeon_bench", "copies/other", 25, Verdict::Ok),
            entry(
                "b2",
                "surgeon_bench",
                copy.to_str().unwrap(),
                15,
                Verdict::Ok,
            ),
            entry("b3", "surgeon_bench", "copies/c1", 200, Verdict::Ok),
            entry("x", "surgeon_inspect", "copies/c1", 26, Verdict::Ok),
        ];
        let restore = entry("r", TOOL, copy.to_str().unwrap(), 100, Verdict::Ok);

        let jobs = followups(project.path(), &entries, &restore);

        let labels: Vec<_> = jobs.iter().map(|j| j.label.as_str()).collect();
        assert_eq!(labels, ["g2", "b2"]);
        assert_eq!(jobs[1].args, entries[4].params);
        assert!(followups(
            project.path(),
            &entries,
            &entry("r", TOOL, "", 9, Verdict::Ok)
        )
        .is_empty());
    }

    #[test]
    fn the_listing_gives_the_tree_depths_the_front_and_the_head_of_a_copy() {
        let project = tempfile::tempdir().unwrap();
        let copy = project.path().join("copies").join("c1");
        std::fs::create_dir_all(&copy).unwrap();
        std::fs::write(copy.join("model.safetensors"), b"w").unwrap();
        let store = store(project.path());
        let metrics = |gain: f64, drift: f64| {
            BTreeMap::from([("gain".into(), gain), ("drift".into(), drift)])
        };
        let base = store.create(&copy, "base", metrics(0.0, 0.0)).unwrap();
        store.create(&copy, "a", metrics(0.2, 0.1)).unwrap();
        store.create(&copy, "b", metrics(0.1, 0.3)).unwrap();
        store.restore(&base.id, &copy).unwrap();
        store.create(&copy, "c", BTreeMap::new()).unwrap();
        assert!(stamp(project.path()).is_some());

        let listing = load(project.path()).unwrap();

        assert_eq!(depths(&listing), [0, 1, 2, 1]);
        assert_eq!(front(&listing), ["0001", "0002"]);
        assert_eq!(head(&listing, &copy), Some("0004"));
        let job = restore_job(&listing.checkpoints[1].manifest, "restore");
        assert_eq!(job.tool, TOOL);
        assert_eq!(
            (job.args["action"].as_str(), job.args["id"].as_str()),
            (Some("restore"), Some("0002"))
        );
    }
}
