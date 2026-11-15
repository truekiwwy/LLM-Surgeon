//! The studio state the views draw: the project list and creation, the open project, the two
//! launch forms (analyses, surgery on a copy), the queue, the run being followed and the
//! ledger exports. No drawing here: `ui::views` does it, the work is in the sibling modules.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use serde_json::Value;

use super::bus::{RunView, EVENTS_FILE, FRAME};
use super::estimate::{self, Estimate};
use super::exec::EngineExecutor;
use super::forms::{self, Form};
use super::jobs::{Budgets, Ended, JobSpec, Manager, COPY_TOOL, OUT};
use super::ledger;
use super::lock;
use super::project::{Manifest, Project};
use crate::platform::Picker;

/// What the window does for this workspace.
pub enum Action {
    /// Open a `graph.json` / `graph.ndjson` without a project (ephemeral).
    OpenGraph(PathBuf),
    /// Show the run of `runs/<id>/` in the 3D view: live while it runs, else replayed from
    /// its `events.ndjson`, over the graph it wrote if any.
    OpenRun {
        dir: PathBuf,
        live: bool,
        target: Target,
    },
}

/// Where a run opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Graph,
    Diff,
    /// The node the run reached last, in the inspector.
    Inspector,
}

/// The tools that write a model: they belong to the surgery form of the Atelier, the others
/// to Analyses. A tool the engine adds later lands in Analyses.
pub const SURGERY_TOOLS: [&str; 5] = [
    "surgeon_apply",
    "surgeon_extract",
    "surgeon_autotune",
    "surgeon_lora",
    "surgeon_diff",
];

pub fn is_surgery(tool: &str) -> bool {
    SURGERY_TOOLS.contains(&tool)
}

/// A project name for a source model: a folder's name, a file's name without its extension,
/// or the folder of a file named only `model` (`Qwen2.5-0.5B/model.safetensors`).
pub fn project_name(source: &Path) -> String {
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    let stem = source.file_stem().map(|s| s.to_string_lossy().into_owned());
    let named = match (source.is_dir(), stem.as_deref()) {
        (true, _) => name(source),
        (false, Some("model")) => source.parent().and_then(name),
        (false, _) => stem,
    };
    named.unwrap_or_default()
}

/// One launch form: the tool picked, what was typed, and the estimate of those arguments.
#[derive(Default)]
pub struct Launch {
    /// Index in [`Studio::forms`].
    pub tool: usize,
    pub values: BTreeMap<String, String>,
    pub label: String,
    pub priority: i32,
    pub max_wall: String,
    pub max_candidates: String,
    /// The preview run in progress and the key of the arguments it estimates.
    pub estimating: Option<(String, String)>,
    /// The last estimate and the key of its arguments: it holds only while they stay.
    pub estimate: Option<(String, Estimate)>,
}

/// What the form can do with what is typed in it right now.
pub struct Prepared {
    pub args: Result<Value, String>,
    /// Identifies these exact arguments.
    pub key: String,
    /// The arguments of the tool's own preview mode, when it has one.
    pub preview: Option<Value>,
}

impl Launch {
    pub fn prepare(&self, form: &Form) -> Prepared {
        let args = form.args(&self.values);
        let key = args
            .as_ref()
            .map(|a| format!("{} {a}", form.tool))
            .unwrap_or_default();
        let preview = args
            .as_ref()
            .ok()
            .filter(|a| !estimate::is_preview(&form.tool, a))
            .and_then(|a| estimate::preview_args(&form.tool, a));
        Prepared { args, key, preview }
    }

    /// The estimate of these exact arguments, if one was made.
    pub fn current(&self, key: &str) -> Option<&Estimate> {
        self.estimate
            .as_ref()
            .filter(|(k, _)| k == key)
            .map(|(_, e)| e)
    }

    pub fn max_wall_secs(&self) -> Option<u64> {
        self.max_wall.trim().parse().ok()
    }
}

/// The working copy the forms target: the newest folder of `copies/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveCopy {
    /// `copies/<id>`, relative to the project.
    pub path: String,
    /// Successful writes the ledger records on it.
    pub writes: usize,
}

pub struct Studio {
    pub workspace: PathBuf,
    pub projects: Vec<(PathBuf, Result<Manifest, String>)>,
    pub new_name: String,
    pub new_source: String,
    creating: Option<Receiver<Result<Project, String>>>,
    /// The system picker of the source model.
    picker: Picker,
    pub manager: Option<Manager>,
    /// The open project's source model held read-only (`None`: no project, or no source).
    pub source_lock: Option<Result<lock::SourceLock, String>>,
    pub forms: Vec<Form>,
    pub analysis: Launch,
    pub surgery: Launch,
    /// The run shown under the queue: a live one, or one replayed from its file.
    pub followed: Option<String>,
    replayed: Option<(String, Result<RunView, String>)>,
    pub graph_path: String,
    pub message: Option<(String, bool)>,
}

impl Studio {
    pub fn new(workspace: PathBuf) -> Self {
        let forms = forms::forms();
        let first = |surgery: bool| {
            forms
                .iter()
                .position(|f| is_surgery(&f.tool) == surgery)
                .unwrap_or(0)
        };
        let mut studio = Self {
            projects: Vec::new(),
            workspace,
            new_name: String::new(),
            new_source: String::new(),
            creating: None,
            picker: Picker::default(),
            manager: None,
            source_lock: None,
            analysis: Launch {
                tool: first(false),
                ..Default::default()
            },
            surgery: Launch {
                tool: first(true),
                ..Default::default()
            },
            forms,
            followed: None,
            replayed: None,
            graph_path: String::new(),
            message: None,
        };
        studio.refresh();
        studio.reset_forms();
        studio
    }

    pub fn refresh(&mut self) {
        self.projects = Project::list(&self.workspace);
    }

    pub fn creating(&self) -> bool {
        self.creating.is_some()
    }

    pub fn picking(&self) -> bool {
        self.picker.busy()
    }

    /// Opens the system picker of the source model (a file, or with `folder` a folder); the
    /// answer fills the creation form.
    pub fn pick_source(&mut self, ctx: &egui::Context, folder: bool, title: &str) {
        let wake = ctx.clone();
        self.picker
            .open(folder, title, move || wake.request_repaint());
    }

    /// The picked model goes in the form, and names the project when no name was typed.
    fn take_source(&mut self, path: &Path) {
        self.new_source = path.display().to_string();
        if self.new_name.trim().is_empty() {
            self.new_name = project_name(path);
        }
    }

    /// Advances the jobs; keeps the window repainting at 60 Hz while one runs. Returns the
    /// runs that just ended (the agent hub waits for its own).
    pub fn tick(&mut self, ctx: &egui::Context) -> Vec<Ended> {
        if let Some(path) = self.picker.poll() {
            self.take_source(&path);
        }
        if let Some(rx) = &self.creating {
            if let Ok(result) = rx.try_recv() {
                self.creating = None;
                match result {
                    Ok(p) => {
                        self.open(ctx, &p.dir);
                        self.refresh();
                    }
                    Err(e) => self.message = Some((e, true)),
                }
            }
        }
        let Some(m) = &mut self.manager else {
            return Vec::new();
        };
        let all = m.tick(Instant::now());
        for ended in &all {
            let line = format!("{} · {} : {:?}", ended.id, ended.spec.label, ended.verdict);
            let bad = ended.verdict != ledger::Verdict::Ok;
            self.message = Some((line, bad));
            for launch in [&mut self.analysis, &mut self.surgery] {
                if let Some((_, key)) = launch
                    .estimating
                    .take_if(|(id, _)| *id == ended.id)
                    .filter(|_| !bad)
                {
                    let e = estimate::read(&ended.envelope);
                    self.message = Some((format!("estimation : {}", e.summary()), false));
                    launch.estimate = Some((key, e));
                }
            }
            if self.followed.as_deref() == Some(ended.id.as_str()) {
                self.replayed = Some((ended.id.clone(), Ok(ended.view.clone())));
            }
        }
        if m.busy() {
            ctx.request_repaint_after(FRAME);
        }
        all
    }

    /// The run the 3D view follows live: the followed one if it runs, else the first running.
    pub fn live_view(&self) -> Option<&RunView> {
        let running = &self.manager.as_ref()?.running;
        running
            .iter()
            .find(|r| self.followed.as_deref() == Some(r.id.as_str()))
            .or(running.first())
            .map(|r| &r.view)
    }

    /// Opens the project of `dir`, unless a job of the current one still runs.
    pub fn open_project(&mut self, ctx: &egui::Context, dir: &Path) {
        if self.manager.as_ref().is_some_and(Manager::busy) {
            self.message = Some(("un job tourne : attendez ou annulez-le".into(), true));
        } else {
            self.open(ctx, dir);
        }
    }

    fn open(&mut self, ctx: &egui::Context, dir: &Path) {
        let wake = ctx.clone();
        match Manager::open(
            dir,
            Arc::new(EngineExecutor::default()),
            Arc::new(move || wake.request_repaint()),
        ) {
            Ok(m) => {
                // The previous project's lock goes with it; this one's holds until the next.
                self.source_lock = m
                    .project
                    .manifest
                    .source
                    .as_ref()
                    .map(|s| lock::lock(&s.path));
                self.manager = Some(m);
                self.followed = None;
                self.replayed = None;
                self.reset_forms();
            }
            Err(e) => self.message = Some((e, true)),
        }
    }

    /// Creates the project typed in `new_name` / `new_source`, hashing the model off the
    /// UI thread, then opens it.
    pub fn create_project(&mut self, ctx: &egui::Context) {
        let (tx, rx) = std::sync::mpsc::channel();
        let (ws, name) = (self.workspace.clone(), self.new_name.trim().to_owned());
        let source =
            Some(PathBuf::from(self.new_source.trim())).filter(|p| !p.as_os_str().is_empty());
        let wake = ctx.clone();
        std::thread::spawn(move || {
            let created = std::fs::create_dir_all(&ws)
                .map_err(|e| e.to_string())
                .and_then(|()| Project::create(&ws, &name, source.as_deref()));
            let _ = tx.send(created);
            wake.request_repaint();
        });
        self.creating = Some(rx);
    }

    /// Puts the two forms back on the defaults of the open project.
    pub fn reset_forms(&mut self) {
        for surgery in [false, true] {
            let tool = self.launch(surgery).tool;
            self.select_tool(surgery, tool);
        }
    }

    pub fn launch(&self, surgery: bool) -> &Launch {
        match surgery {
            true => &self.surgery,
            false => &self.analysis,
        }
    }

    pub fn launch_mut(&mut self, surgery: bool) -> &mut Launch {
        match surgery {
            true => &mut self.surgery,
            false => &mut self.analysis,
        }
    }

    /// Picks the tool of a form and prefills it for the open project.
    pub fn select_tool(&mut self, surgery: bool, tool: usize) {
        let values = match (self.forms.get(tool), &self.manager) {
            (Some(form), Some(m)) => default_values(form, &m.project),
            _ => BTreeMap::new(),
        };
        let label = self
            .forms
            .get(tool)
            .map(|f| f.title.clone())
            .unwrap_or_default();
        let launch = self.launch_mut(surgery);
        launch.tool = tool;
        launch.values = values;
        launch.label = label;
    }

    /// A working copy of the source model, as a job (`copies/<id>/`).
    /// Queues a working copy; its run id, or `None` with the reason in `message`.
    pub fn submit_copy(&mut self) -> Option<String> {
        let m = self.manager.as_mut()?;
        let spec = JobSpec {
            tool: COPY_TOOL.into(),
            label: "copie de travail".into(),
            args: serde_json::json!({}),
            priority: 0,
            budgets: Budgets::default(),
            resumed_from: None,
            read_only: false,
        };
        m.submit(spec)
            .map_err(|e| self.message = Some((e, true)))
            .ok()
    }

    /// Runs the tool's own preview mode (estimate, dry_run, plan): nothing is written.
    pub fn request_estimate(&mut self, surgery: bool) {
        let launch = self.launch(surgery);
        let Some(form) = self.forms.get(launch.tool) else {
            return;
        };
        let prepared = launch.prepare(form);
        let label = format!("estimation · {}", launch.label);
        let tool = form.tool.clone();
        let (Some(preview), Some(m)) = (prepared.preview, &mut self.manager) else {
            return;
        };
        let submitted = m.submit(JobSpec {
            tool,
            label,
            args: preview,
            priority: 10,
            budgets: Budgets::default(),
            resumed_from: None,
            read_only: false,
        });
        match submitted {
            Ok(id) => self.launch_mut(surgery).estimating = Some((id, prepared.key)),
            Err(e) => self.message = Some((e, true)),
        }
    }

    /// Queues the form as typed, after the budget gate.
    pub fn queue(&mut self, surgery: bool) {
        let launch = self.launch(surgery);
        let Some(form) = self.forms.get(launch.tool) else {
            return;
        };
        let prepared = launch.prepare(form);
        let max_wall = launch.max_wall_secs();
        let gate = estimate::gate(
            prepared.preview.is_some(),
            max_wall,
            launch.current(&prepared.key),
        );
        let spec = prepared.args.and_then(|args| {
            gate?;
            Ok(JobSpec {
                tool: form.tool.clone(),
                label: launch.label.clone(),
                args,
                priority: launch.priority,
                budgets: Budgets {
                    max_wall_time_secs: max_wall,
                    max_candidates: launch.max_candidates.trim().parse().ok(),
                },
                resumed_from: None,
                read_only: false,
            })
        });
        let Some(m) = &mut self.manager else {
            return;
        };
        match spec.and_then(|spec| m.submit(spec)) {
            Ok(id) => {
                self.followed = Some(id.clone());
                self.message = Some((format!("{id} en file"), false));
            }
            Err(e) => self.message = Some((e, true)),
        }
    }

    /// The followed run: live, replayed from its file, or not started yet (`None`).
    pub fn followed_view(&mut self) -> Option<Result<&RunView, &str>> {
        let m = self.manager.as_ref()?;
        let id = self.followed.as_ref()?;
        if let Some(run) = m.running.iter().find(|r| &r.id == id) {
            return Some(Ok(&run.view));
        }
        let path = m.project.runs_dir().join(id).join(EVENTS_FILE);
        if self.replayed.as_ref().is_none_or(|(rid, _)| rid != id) && path.exists() {
            self.replayed = Some((id.clone(), RunView::replay(&path)));
        }
        self.replayed
            .as_ref()
            .filter(|(rid, _)| rid == id)
            .map(|(_, view)| view.as_ref().map_err(String::as_str))
    }

    /// Writes the ledger next to the project as `ledger.json` or `ledger.md`.
    pub fn export_ledger(&mut self, markdown: bool) {
        let Some(m) = &self.manager else {
            return;
        };
        let (file, render): (_, fn(&[ledger::Entry]) -> String) = match markdown {
            true => ("ledger.md", ledger::to_markdown),
            false => ("ledger.json", ledger::to_json),
        };
        let path = m.project.dir.join(file);
        let written = ledger::read(&m.project.ledger_path())
            .and_then(|entries| std::fs::write(&path, render(&entries)).map_err(|e| e.to_string()));
        self.message = Some(match written {
            Ok(()) => (format!("journal exporté : {}", path.display()), false),
            Err(e) => (e, true),
        });
    }

    /// The working copy in use and how many writes the ledger records on it.
    pub fn active_copy(&self, entries: &[ledger::Entry]) -> Option<ActiveCopy> {
        let path = newest_copy(&self.manager.as_ref()?.project)?;
        let writes = entries.iter().filter(|e| is_write_on(e, &path)).count();
        Some(ActiveCopy { path, writes })
    }
}

/// Whether the ledger entry is a successful write on the copy `path`: a surgery tool that
/// ran for real (no preview mode) on that model.
pub fn is_write_on(entry: &ledger::Entry, path: &str) -> bool {
    entry.verdict == ledger::Verdict::Ok
        && is_surgery(&entry.tool)
        && !estimate::is_preview(&entry.tool, &entry.params)
        && entry.params["model"] == path
        // Extraction and diff read the model: only these three change it.
        && matches!(
            entry.tool.as_str(),
            "surgeon_apply" | "surgeon_lora" | "surgeon_autotune"
        )
}

/// `copies/<id>` of the newest finished working copy.
pub fn newest_copy(project: &Project) -> Option<String> {
    newest_copy_in(&project.copies_dir()).map(|id| format!("copies/{id}"))
}

/// The `<id>` of the newest finished copy in a project's `copies` folder.
pub fn newest_copy_in(copies: &Path) -> Option<String> {
    std::fs::read_dir(copies)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().ends_with(".partial"))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
}

/// Prefilled values of a form: outputs in the run folder, the model on the newest working
/// copy (or the source when there is none).
pub fn default_values(form: &Form, project: &Project) -> BTreeMap<String, String> {
    let model = newest_copy(project).or_else(|| {
        project
            .manifest
            .source
            .as_ref()
            .map(|s| s.path.display().to_string())
    });
    let out_file = match form.tool.as_str() {
        "surgeon_extract" => "direction.safetensors",
        "surgeon_bench" => "logits.safetensors",
        "surgeon_autotune" => "patch.json",
        "surgeon_diff" => "model.surgery",
        "surgeon_lora" => "adapter_model.safetensors",
        _ => "result",
    };
    let mut values = BTreeMap::new();
    for field in &form.fields {
        let value = match field.name.as_str() {
            "model" => model.clone(),
            "out" => Some(format!("{OUT}/{out_file}")),
            "out_dir" => Some(OUT.into()),
            "job_dir" => Some(format!("{OUT}/job")),
            _ => None,
        };
        if let Some(v) = value {
            values.insert(field.name.clone(), v);
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picked_model_names_its_project() {
        let ws = tempfile::tempdir().unwrap();
        let dir = ws.path().join("Qwen2.5-0.5B");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(project_name(&dir), "Qwen2.5-0.5B");
        assert_eq!(project_name(&dir.join("model.safetensors")), "Qwen2.5-0.5B");
        assert_eq!(
            project_name(&dir.join("llama-3-8b.Q4_K_M.gguf")),
            "llama-3-8b.Q4_K_M"
        );
    }

    #[test]
    fn outputs_default_to_the_run_folder_and_the_model_to_the_newest_copy() {
        let ws = tempfile::tempdir().unwrap();
        let model = ws.path().join("m.gguf");
        std::fs::write(&model, b"w").unwrap();
        let p = Project::create(ws.path(), "p", Some(&model)).unwrap();
        let forms = forms::forms();
        let form = |t: &str| forms.iter().find(|f| f.tool == t).unwrap();
        let v = default_values(form("surgeon_extract"), &p);
        assert_eq!(v["out"], "{out}/direction.safetensors");
        assert_eq!(
            v["model"],
            p.manifest
                .source
                .as_ref()
                .unwrap()
                .path
                .display()
                .to_string()
        );
        std::fs::create_dir_all(p.copies_dir().join("c1")).unwrap();
        std::fs::create_dir_all(p.copies_dir().join("c2.partial")).unwrap();
        let v = default_values(form("surgeon_graph"), &p);
        assert_eq!(v["model"], "copies/c1");
        assert_eq!(v["out_dir"], OUT);
        assert!(form("surgeon_graph").args(&v).is_ok());
    }

    #[test]
    fn every_engine_tool_is_in_exactly_one_form_and_the_surgery_ones_exist() {
        let forms = forms::forms();
        for tool in SURGERY_TOOLS {
            assert!(forms.iter().any(|f| f.tool == tool), "{tool}");
        }
        let studio = Studio::new(tempfile::tempdir().unwrap().path().to_path_buf());
        assert!(is_surgery(&studio.forms[studio.surgery.tool].tool));
        assert!(!is_surgery(&studio.forms[studio.analysis.tool].tool));
    }

    #[test]
    fn a_copy_counts_only_the_real_writes_the_ledger_records_on_it() {
        let entry = |tool: &str, params: Value, verdict| ledger::Entry {
            run_id: "r".into(),
            tool: tool.into(),
            label: String::new(),
            params,
            resumed_from: None,
            started_unix: 0,
            finished_unix: 1,
            hashes: Default::default(),
            versions: Default::default(),
            verdict,
            exit_code: 0,
            result: Value::Null,
            error: None,
        };
        let on_c1 = serde_json::json!({"model": "copies/c1", "patch": "p.json"});
        let ok = ledger::Verdict::Ok;
        assert!(is_write_on(
            &entry("surgeon_apply", on_c1.clone(), ok),
            "copies/c1"
        ));
        assert!(!is_write_on(
            &entry("surgeon_apply", on_c1.clone(), ok),
            "copies/c2"
        ));
        assert!(!is_write_on(
            &entry("surgeon_apply", on_c1.clone(), ledger::Verdict::Failed),
            "copies/c1"
        ));
        let dry = serde_json::json!({"model": "copies/c1", "patch": "p.json", "dry_run": true});
        assert!(!is_write_on(&entry("surgeon_apply", dry, ok), "copies/c1"));
        assert!(!is_write_on(
            &entry("surgeon_extract", on_c1.clone(), ok),
            "copies/c1"
        ));
        assert!(!is_write_on(
            &entry("surgeon_graph", on_c1, ok),
            "copies/c1"
        ));
    }
}
