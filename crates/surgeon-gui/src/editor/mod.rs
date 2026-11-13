//! The `patch.json` editor: the text is parsed and validated by the library itself
//! (`patch::parse_patch`, `patch::validate_patch`) a short while after the last keystroke,
//! every issue is brought back to the line its JSON path points at, and the layer and
//! projection selectors that already parse are turned into the 3D nodes they will touch,
//! before anything is written.

pub mod view;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::text::{LayoutJob, TextFormat};
use eframe::egui::{Color32, FontId};
use weight_surgeon::graph::output::GraphReport;
use weight_surgeon::graph::GraphRole;
use weight_surgeon::patch::{
    parse_patch, validate_patch, LayerSelection, PatchError, Projection, ValidationOptions,
};

use crate::runner::{Step, Task};
use crate::theme;

/// Time without a keystroke before the text is validated again.
pub const DEBOUNCE: Duration = Duration::from_millis(250);

/// The text a new editor starts with: a patch that only lacks its vector file.
pub const TEMPLATE: &str = r#"{
  "schema_version": 1,
  "description": "",
  "backup": true,
  "operations": [
    {
      "id": "op-1",
      "layers": "all",
      "projections": ["down_proj"],
      "operation": "subtract",
      "vector_file": "direction.safetensors",
      "alpha": 1.0,
      "axis": "cols"
    }
  ]
}
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// 0-based line.
    pub line: usize,
    /// JSON path (`operations[0].alpha`), empty for a syntax error.
    pub path: String,
    pub message: String,
    pub error: bool,
}

/// One selection that parses: the layers and projections of an operation.
pub type Selection = (LayerSelection, Vec<Projection>);

#[derive(Debug, Clone, Default)]
pub struct Checked {
    pub diagnostics: Vec<Diagnostic>,
    /// The selectors that already parse, valid patch or not.
    pub selections: Vec<Selection>,
    /// No error: the patch can be run.
    pub valid: bool,
}

/// Parses then validates `text`; vector files are looked for relative to `base_dir`.
pub fn check(text: &str, base_dir: &Path) -> Checked {
    let failed = |line: usize, message: String| Checked {
        diagnostics: vec![Diagnostic {
            line,
            path: String::new(),
            message,
            error: true,
        }],
        ..Default::default()
    };
    let patch = match parse_patch(text) {
        Ok(patch) => patch,
        Err(PatchError::InvalidJson { message, line, .. }) => {
            return failed(line.saturating_sub(1), message)
        }
        Err(e) => return failed(0, e.to_string()),
    };
    let selections = patch
        .operations
        .iter()
        .filter_map(|op| {
            let layers = op.layers.to_selection().ok()?;
            (!op.projections.is_empty()).then(|| (layers, op.projections.clone()))
        })
        .collect();
    let lines = json_lines(text);
    let diagnostic = |issue: weight_surgeon::patch::ValidationIssue, error: bool| Diagnostic {
        line: locate(&lines, &issue.path),
        path: issue.path,
        message: issue.message,
        error,
    };
    let (diagnostics, valid) = match validate_patch(&patch, base_dir, &ValidationOptions::default())
    {
        Ok(ok) => (
            ok.warnings
                .into_iter()
                .map(|w| diagnostic(w, false))
                .collect(),
            true,
        ),
        Err(PatchError::Invalid(issues)) => (
            issues.into_iter().map(|i| diagnostic(i, true)).collect(),
            false,
        ),
        Err(e) => (failed(0, e.to_string()).diagnostics, false),
    };
    Checked {
        diagnostics,
        selections,
        valid,
    }
}

/// The line (0-based) where each JSON path starts: keys at their key, array items at
/// their first character. Only called on text `serde_json` already accepted.
pub fn json_lines(text: &str) -> HashMap<String, usize> {
    struct Scan<'a> {
        bytes: &'a [u8],
        i: usize,
        line: usize,
        out: HashMap<String, usize>,
    }
    impl Scan<'_> {
        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.i).copied()
        }
        fn blank(&mut self) {
            while let Some(b) = self.peek().filter(u8::is_ascii_whitespace) {
                self.line += usize::from(b == b'\n');
                self.i += 1;
            }
        }
        /// At an opening quote; returns the raw content.
        fn string(&mut self) -> String {
            self.i += 1;
            let start = self.i;
            while let Some(b) = self.peek() {
                match b {
                    b'\\' => self.i += 2,
                    b'"' => break,
                    _ => self.i += 1,
                }
            }
            let end = self.i.min(self.bytes.len());
            self.i += 1;
            String::from_utf8_lossy(&self.bytes[start..end]).into_owned()
        }
        fn value(&mut self, path: String) {
            self.blank();
            self.out.entry(path.clone()).or_insert(self.line);
            match self.peek() {
                Some(b'{') => {
                    self.i += 1;
                    loop {
                        self.blank();
                        match self.peek() {
                            Some(b'"') => {
                                let line = self.line;
                                let key = self.string();
                                let child = match path.is_empty() {
                                    true => key,
                                    false => format!("{path}.{key}"),
                                };
                                self.out.insert(child.clone(), line);
                                self.blank();
                                self.i += 1; // ':'
                                self.value(child);
                            }
                            Some(b',') => self.i += 1,
                            Some(b'}') | None => {
                                self.i += 1;
                                return;
                            }
                            Some(_) => self.i += 1,
                        }
                    }
                }
                Some(b'[') => {
                    self.i += 1;
                    let mut index = 0;
                    loop {
                        self.blank();
                        match self.peek() {
                            Some(b',') => self.i += 1,
                            Some(b']') | None => {
                                self.i += 1;
                                return;
                            }
                            Some(_) => {
                                self.value(format!("{path}[{index}]"));
                                index += 1;
                            }
                        }
                    }
                }
                Some(b'"') => {
                    self.string();
                }
                Some(_) => {
                    let start = self.i;
                    while self.peek().is_some_and(|b| {
                        !matches!(b, b',' | b'}' | b']') && !b.is_ascii_whitespace()
                    }) {
                        self.i += 1;
                    }
                    // Never stall on a stray byte.
                    self.i = self.i.max(start + 1);
                }
                None => {}
            }
        }
    }
    let mut scan = Scan {
        bytes: text.as_bytes(),
        i: 0,
        line: 0,
        out: HashMap::new(),
    };
    scan.value(String::new());
    scan.out
}

/// The line of `path`, or of its nearest existing parent (a missing field is reported on
/// its object), or the first line.
pub fn locate(lines: &HashMap<String, usize>, path: &str) -> usize {
    let mut path = path;
    loop {
        if let Some(&line) = lines.get(path) {
            return line;
        }
        match path.rfind(['.', '[']) {
            Some(cut) => path = &path[..cut],
            None => return 0,
        }
    }
}

/// The graph role a patch projection writes.
fn role_of(projection: Projection) -> GraphRole {
    match projection {
        Projection::Gate => GraphRole::Gate,
        Projection::Up => GraphRole::Up,
        Projection::Down => GraphRole::MlpOut,
        Projection::AttnOut => GraphRole::AttnOut,
        Projection::EmbedTokens => GraphRole::Embed,
    }
}

fn in_layers(selection: &LayerSelection, layer: usize) -> bool {
    match selection {
        LayerSelection::All => true,
        LayerSelection::List(list) => list.contains(&layer),
        LayerSelection::Range {
            start,
            end_inclusive,
        } => (*start..=*end_inclusive).contains(&layer),
    }
}

/// Which nodes of `report` the selections touch, as the executor resolves them: the role
/// of a projection on the selected layers (every expert of a down_proj included), and
/// `embed_tokens` whatever the layers.
pub fn targets(selections: &[Selection], report: &GraphReport) -> Vec<bool> {
    report
        .nodes
        .iter()
        .map(|node| {
            selections.iter().any(|(layers, projections)| {
                projections.iter().any(|&p| {
                    role_of(p) == node.role
                        && (p == Projection::EmbedTokens
                            || node.layer.is_some_and(|l| in_layers(layers, l)))
                })
            })
        })
        .collect()
}

const KEY: Color32 = theme::TEXT;
const STRING: Color32 = Color32::from_rgb(135, 175, 255);
const NUMBER: Color32 = Color32::from_rgb(215, 175, 95);
const LITERAL: Color32 = Color32::from_rgb(95, 175, 255);

/// Sober JSON coloring, never wrapped (rows are lines, the gutter relies on it).
pub fn highlight(text: &str, font: FontId) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.wrap.max_width = f32::INFINITY;
    let bytes = text.as_bytes();
    let mut push = |range: std::ops::Range<usize>, color: Color32| {
        job.append(&text[range], 0.0, TextFormat::simple(font.clone(), color));
    };
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let color = match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' && bytes[i] != b'\n' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
                i = (i + 1).min(bytes.len());
                let next = bytes[i..].iter().find(|b| !b.is_ascii_whitespace());
                match next {
                    Some(b':') => KEY,
                    _ => STRING,
                }
            }
            b'-' | b'0'..=b'9' => {
                while i < bytes.len()
                    && matches!(bytes[i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    i += 1;
                }
                NUMBER
            }
            b'a'..=b'z' => {
                while i < bytes.len() && bytes[i].is_ascii_lowercase() {
                    i += 1;
                }
                LITERAL
            }
            _ => {
                // Punctuation and blanks up to the next token; stops on ASCII, so the
                // slice stays on a char boundary.
                i += 1;
                while i < bytes.len()
                    && !matches!(bytes[i], b'"' | b'-' | b'0'..=b'9' | b'a'..=b'z')
                {
                    i += 1;
                }
                theme::DIM
            }
        };
        // An escape can skip into a multi-byte char: move on to the next boundary.
        i = i.min(bytes.len());
        while !text.is_char_boundary(i) {
            i += 1;
        }
        push(start..i, color);
    }
    job
}

/// The editor buttons that start a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Run {
    DryRun,
    Apply,
}

pub struct Editor {
    pub text: String,
    /// The `patch.json` on disk (import, export, and the file `apply` reads).
    pub file: String,
    /// The model Dry-Run and Apply act on.
    pub model: String,
    pub checked: Checked,
    pub message: Option<(String, bool)>,
    edited: Option<Instant>,
}

impl Default for Editor {
    fn default() -> Self {
        let mut editor = Self {
            text: TEMPLATE.into(),
            file: "patch.json".into(),
            model: String::new(),
            checked: Checked::default(),
            message: None,
            edited: None,
        };
        editor.validate();
        editor
    }
}

impl Editor {
    /// Relative vector files resolve against the directory of the patch file.
    pub fn base_dir(&self) -> PathBuf {
        Path::new(self.file.trim())
            .parent()
            .unwrap_or(Path::new(""))
            .to_path_buf()
    }

    pub fn validate(&mut self) {
        self.checked = check(&self.text, &self.base_dir());
        self.edited = None;
    }

    /// A keystroke: validation waits for [`DEBOUNCE`].
    pub fn touch(&mut self) {
        self.edited = Some(Instant::now());
    }

    /// Validates once the debounce has passed; otherwise how long to wait.
    pub fn tick(&mut self) -> Option<Duration> {
        let wait = DEBOUNCE.checked_sub(self.edited?.elapsed());
        if wait.is_none() {
            self.validate();
        }
        wait
    }

    /// Replaces the text (a card opened in the editor, an import) and validates it now.
    pub fn set_text(&mut self, text: String) {
        self.text = text;
        self.validate();
    }

    pub fn import(&mut self) -> Result<(), String> {
        let path = self.file.trim();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path} : {e}"))?;
        self.set_text(text);
        Ok(())
    }

    pub fn export(&self) -> Result<(), String> {
        let path = self.file.trim();
        if path.is_empty() {
            return Err("aucun fichier patch.json".into());
        }
        std::fs::write(path, &self.text).map_err(|e| format!("{path} : {e}"))
    }

    /// Saves the text and returns the `apply` step. Apply is refused without a backup:
    /// the studio only writes a model it can restore.
    pub fn step(&mut self, run: Run) -> Result<Step, String> {
        self.validate();
        if !self.checked.valid {
            return Err("le patch a des erreurs : corrigez-les avant de l'exécuter".into());
        }
        let model = self.model.trim();
        if model.is_empty() {
            return Err("aucun modèle : renseignez le champ modèle".into());
        }
        if run == Run::Apply && parse_patch(&self.text).is_ok_and(|p| !p.backup) {
            return Err("Apply exige \"backup\": true (sinon pas de Rollback possible)".into());
        }
        self.export()?;
        let mut args = vec![
            "apply".into(),
            "--model".into(),
            model.into(),
            "--patch".into(),
            self.file.trim().into(),
        ];
        let label = match run {
            Run::DryRun => {
                args.push("--dry-run".into());
                "éditeur : dry-run"
            }
            Run::Apply => "éditeur : apply",
        };
        Ok(Step {
            card: None,
            label: label.into(),
            task: Task::Cli(args),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::backups_of;
    use crate::runner::Job;
    use weight_surgeon::graph::stream::GraphPatch;

    fn report() -> GraphReport {
        let line = include_str!("../../tests/fixtures/qwen2-mini.ndjson");
        match serde_json::from_str(line.trim()).unwrap() {
            GraphPatch::Snapshot { report } => *report,
            GraphPatch::Update { .. } => unreachable!(),
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("surgeon-gui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn json_paths_map_to_their_lines() {
        let lines = json_lines(TEMPLATE);
        assert_eq!(lines["schema_version"], 1);
        assert_eq!(lines["operations"], 4);
        assert_eq!(lines["operations[0]"], 5);
        assert_eq!(lines["operations[0].alpha"], 11);
        assert_eq!(lines["operations[0].projections[0]"], 8);
        assert_eq!(locate(&lines, "operations[0].alpha"), 11);
        // A missing field is reported on its operation, an unknown path on line 0.
        assert_eq!(locate(&lines, "operations[0].vector_tensor_b"), 5);
        assert_eq!(locate(&lines, "nowhere"), 0);
    }

    #[test]
    fn issues_land_on_the_line_of_their_json_path() {
        let dir = temp_dir("check");
        // The template's vector is missing: one error, on the vector_file line.
        let checked = check(TEMPLATE, &dir);
        assert!(!checked.valid);
        let [d] = &checked.diagnostics[..] else {
            panic!("{:?}", checked.diagnostics)
        };
        assert_eq!((d.line, d.path.as_str()), (10, "operations[0].vector_file"));
        // The selector already parses: the 3D preview does not wait for the vector.
        assert_eq!(checked.selections.len(), 1);

        std::fs::write(dir.join("direction.safetensors"), b"").unwrap();
        let strong = TEMPLATE.replace("\"alpha\": 1.0", "\"alpha\": 7.5");
        let checked = check(&strong, &dir);
        assert!(checked.valid, "{:?}", checked.diagnostics);
        assert_eq!(checked.diagnostics[0].line, 11);
        assert!(!checked.diagnostics[0].error, "a strong alpha is a warning");

        let broken = TEMPLATE.replace("\"axis\": \"cols\"", "\"axis\": \"cols\",");
        let checked = check(&broken, &dir);
        assert!(!checked.valid && checked.selections.is_empty());
        assert_eq!(checked.diagnostics[0].line, 13, "{:?}", checked.diagnostics);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_debounce_holds_validation_until_typing_stops() {
        let mut editor = Editor {
            text: "{".into(),
            ..Default::default()
        };
        editor.touch();
        assert!(editor.tick().is_some(), "still typing");
        assert!(
            editor.checked.selections.len() == 1,
            "the last check is kept"
        );
        std::thread::sleep(DEBOUNCE);
        assert_eq!(editor.tick(), None);
        assert!(!editor.checked.valid && editor.checked.diagnostics[0].path.is_empty());
        assert_eq!(editor.tick(), None, "nothing pending");
    }

    #[test]
    fn selectors_light_the_nodes_they_will_touch() {
        let report = report();
        let pick = |layers: &str, projections: &[Projection]| {
            let selection = (layers.parse().unwrap(), projections.to_vec());
            let hit = targets(&[selection], &report);
            report
                .nodes
                .iter()
                .zip(hit)
                .filter(|(_, h)| *h)
                .map(|(n, _)| n.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(pick("1", &[Projection::AttnOut]), ["L1.attn_out"]);
        let downs = pick("all", &[Projection::Down]);
        assert!(!downs.is_empty());
        assert!(downs.iter().all(|id| id.ends_with("mlp_out")), "{downs:?}");
        let embed = pick("99", &[Projection::EmbedTokens]);
        assert_eq!(embed.len(), 1, "embed_tokens ignores the layers: {embed:?}");
        assert!(pick("99", &[Projection::Down]).is_empty());
    }

    #[test]
    fn highlighting_keeps_every_byte() {
        let text = "{\n  \"clé\": [1.5e-3, true, \"vé\"] }\n";
        let job = highlight(text, FontId::monospace(12.0));
        assert_eq!(job.text, text);
        let colors: Vec<Color32> = job.sections.iter().map(|s| s.format.color).collect();
        // `{\n  `, `"clé"`, `: [`, `1.5e-3`, `, `, `true`, `, `, `"vé"`, `] }\n`.
        let expected = [
            theme::DIM,
            KEY,
            theme::DIM,
            NUMBER,
            theme::DIM,
            LITERAL,
            theme::DIM,
            STRING,
            theme::DIM,
        ];
        assert_eq!(colors, expected);
    }

    /// Single-file model, `gate [2,3] = 1..6`, as `tests/integration.rs` writes it.
    fn tiny_model(dir: &Path) -> PathBuf {
        let values: Vec<u8> = (1..=6).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let mut header = serde_json::json!({ "model.layers.0.mlp.gate_proj.weight":
            {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, values.len()]} })
        .to_string();
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header.as_bytes());
        file.extend(values);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, file).unwrap();
        path
    }

    #[test]
    fn dry_run_then_apply_then_rollback_on_a_tiny_model() {
        let dir = temp_dir("apply");
        let model = tiny_model(&dir);
        let original = std::fs::read(&model).unwrap();
        std::fs::write(dir.join("v3.json"), "[1, 1, 1]").unwrap();
        let mut editor = Editor {
            file: dir.join("patch.json").display().to_string(),
            model: model.display().to_string(),
            ..Default::default()
        };
        editor.set_text(
            TEMPLATE
                .replace("down_proj", "gate_proj")
                .replace("direction.safetensors", "v3.json")
                .replace("\"cols\"", "\"rows\"")
                .replace("\"all\"", "\"0\""),
        );
        assert!(editor.checked.valid, "{:?}", editor.checked.diagnostics);

        let done = Job::spawn(vec![editor.step(Run::DryRun).unwrap()], || {}).wait();
        assert_eq!(done[0].code, 0, "{}", done[0].envelope);
        assert_eq!(done[0].envelope["data"]["report"]["dry_run"], true);
        assert_eq!(
            std::fs::read(&model).unwrap(),
            original,
            "a dry run writes nothing"
        );

        let done = Job::spawn(vec![editor.step(Run::Apply).unwrap()], || {}).wait();
        assert_eq!(done[0].code, 0, "{}", done[0].envelope);
        assert_ne!(std::fs::read(&model).unwrap(), original);
        let backups = backups_of(&done[0]);
        assert_eq!(backups.len(), 1);

        let restore = Step {
            card: None,
            label: "rollback".into(),
            task: Task::Restore(backups),
        };
        let done = Job::spawn(vec![restore], || {}).wait();
        assert_eq!(done[0].code, 0, "{}", done[0].envelope);
        assert_eq!(
            std::fs::read(&model).unwrap(),
            original,
            "restored byte for byte"
        );

        editor.set_text(editor.text.replace("\"backup\": true", "\"backup\": false"));
        assert!(editor.step(Run::Apply).unwrap_err().contains("backup"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
