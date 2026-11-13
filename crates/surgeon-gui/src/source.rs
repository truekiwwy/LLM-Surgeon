//! Where the graph comes from: a `graph.json` (one state), or an NDJSON stream of
//! [`GraphPatch`] lines (`surgeon graph --format ndjson`, then whatever an orchestrator or an
//! agent appends), replayed into a history that the timeline scrubs.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use weight_surgeon::graph::diff::NodeDiff;
use weight_surgeon::graph::output::{GraphReport, SCHEMA_VERSION};
use weight_surgeon::graph::stream::GraphPatch;

/// How often a followed file is polled once its end is reached.
const POLL: Duration = Duration::from_millis(200);

/// What the command line points at.
pub enum Input {
    Report(Box<GraphReport>),
    /// `follow`: keep reading after the end (a file another process appends to).
    Stream {
        reader: Box<dyn BufRead + Send>,
        follow: bool,
    },
}

/// `-` is NDJSON on stdin, `*.ndjson` a followed stream file, anything else a `graph.json`.
pub fn open(path: &Path) -> Result<Input, String> {
    if path.as_os_str() == "-" {
        return Ok(Input::Stream {
            reader: Box::new(BufReader::new(std::io::stdin())),
            follow: false,
        });
    }
    let fail = |e: std::io::Error| format!("{} : {e}", path.display());
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("ndjson"))
    {
        return Ok(Input::Stream {
            reader: Box::new(BufReader::new(File::open(path).map_err(fail)?)),
            follow: true,
        });
    }
    let bytes = std::fs::read(path).map_err(fail)?;
    parse_report(&bytes)
        .map(|r| Input::Report(Box::new(r)))
        .map_err(|e| format!("{} : {e}", path.display()))
}

fn parse_report(bytes: &[u8]) -> Result<GraphReport, String> {
    let report: GraphReport =
        serde_json::from_slice(bytes).map_err(|e| format!("graph.json illisible ({e})"))?;
    match report.schema_version == SCHEMA_VERSION {
        true => Ok(report),
        false => Err(format!(
            "schema_version {} (attendu {SCHEMA_VERSION})",
            report.schema_version
        )),
    }
}

/// Reads patches on a thread, one per line; `wake` asks the UI to repaint. Malformed lines
/// are reported and skipped, never fatal. A followed stream waits for complete lines.
pub fn spawn_reader(
    mut reader: Box<dyn BufRead + Send>,
    follow: bool,
    wake: impl Fn() + Send + 'static,
) -> Receiver<Result<GraphPatch, String>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        loop {
            let read = match reader.read_line(&mut line) {
                Ok(read) => read,
                Err(e) => {
                    let _ = tx.send(Err(format!("lecture du flux : {e}")));
                    wake();
                    return;
                }
            };
            if follow && (read == 0 || !line.ends_with('\n')) {
                // At the end, or a line still being written: the next read appends to it.
                std::thread::sleep(POLL);
                continue;
            }
            let text = line.trim();
            if !text.is_empty() {
                let patch = serde_json::from_str(text).map_err(|e| format!("ligne NDJSON : {e}"));
                if tx.send(patch).is_err() {
                    return;
                }
                wake();
            }
            line.clear();
            if read == 0 {
                return;
            }
        }
    });
    rx
}

/// One state of the history.
#[derive(Debug, Clone)]
pub struct Step {
    pub report: GraphReport,
    /// Nodes the patch that made this state touched (changed or moved), with their delta
    /// against the previous state when their analysis changed.
    pub touched: BTreeMap<String, Option<NodeDiff>>,
    /// `(edge, source node, target node)` indices of every edge whose ends exist.
    pub edges: Vec<(usize, usize, usize)>,
}

impl Step {
    fn new(report: GraphReport, touched: BTreeMap<String, Option<NodeDiff>>) -> Self {
        let index: HashMap<&str, usize> = report
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();
        let edges = report
            .edges
            .iter()
            .enumerate()
            .filter_map(|(k, e)| {
                Some((
                    k,
                    *index.get(e.source.as_str())?,
                    *index.get(e.target.as_str())?,
                ))
            })
            .collect();
        Self {
            report,
            touched,
            edges,
        }
    }
}

/// Every state of a stream, oldest first, and the one on screen.
///
/// shortcut: keeps a full copy of the report per patch; fine for hundreds of patches on a
/// graph of a few thousand nodes, switch to periodic keyframes plus replay if streams grow.
#[derive(Debug, Default)]
pub struct Timeline {
    steps: Vec<Step>,
    cursor: usize,
}

impl Timeline {
    pub fn from_report(report: GraphReport) -> Self {
        Self {
            steps: vec![Step::new(report, BTreeMap::new())],
            cursor: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The cursor is on the newest state (new patches move it along).
    pub fn is_live(&self) -> bool {
        self.cursor + 1 >= self.steps.len()
    }

    pub fn seek(&mut self, cursor: usize) {
        self.cursor = cursor.min(self.steps.len().saturating_sub(1));
    }

    pub fn current(&self) -> Option<&Step> {
        self.steps.get(self.cursor)
    }

    /// Appends the state `patch` makes from the newest one. An update before any snapshot,
    /// or one naming an unknown node, is refused and leaves the history as it was.
    pub fn push(&mut self, patch: &GraphPatch) -> Result<(), String> {
        let live = self.is_live();
        let step = match (patch, self.steps.last()) {
            (GraphPatch::Snapshot { report }, _) => Step::new((**report).clone(), BTreeMap::new()),
            (GraphPatch::Update { .. }, None) => {
                return Err("mise à jour reçue avant le premier snapshot".into())
            }
            (GraphPatch::Update { nodes, moves, .. }, Some(last)) => {
                let mut report = last.report.clone();
                patch.apply(&mut report).map_err(|e| e.to_string())?;
                let touched = nodes
                    .iter()
                    .map(|u| (u.node.id.clone(), Some(u.diff.clone())))
                    .chain(moves.iter().map(|m| (m.id.clone(), None)))
                    .collect();
                Step::new(report, touched)
            }
        };
        self.steps.push(step);
        if live {
            self.cursor = self.steps.len() - 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const FIXTURE: &str = include_str!("../tests/fixtures/qwen2-mini.ndjson");

    fn snapshot() -> GraphReport {
        match serde_json::from_str(FIXTURE.trim()).unwrap() {
            GraphPatch::Snapshot { report } => *report,
            GraphPatch::Update { .. } => panic!("the fixture is a snapshot"),
        }
    }

    #[test]
    fn a_report_resolves_its_edges_and_checks_its_schema() {
        let report = snapshot();
        let step = Step::new(report.clone(), BTreeMap::new());
        assert_eq!(step.edges.len(), report.edges.len());
        let text = serde_json::to_vec(&report).unwrap();
        assert_eq!(parse_report(&text).unwrap(), report);
        let mut old = report;
        old.schema_version = 0;
        let err = parse_report(&serde_json::to_vec(&old).unwrap()).unwrap_err();
        assert!(err.contains("schema_version 0"), "{err}");
    }

    #[test]
    fn the_timeline_replays_patches_and_follows_only_when_live() {
        let before = snapshot();
        let mut after = before.clone();
        after.nodes[3].metrics.frobenius_norm *= 0.5;
        let changed = after.nodes[3].id.clone();

        let mut timeline = Timeline::default();
        let update = GraphPatch::between(&before, &after);
        assert!(timeline.push(&update).is_err(), "no state to update yet");
        assert!(timeline.is_empty());

        timeline.push(&GraphPatch::snapshot(&before)).unwrap();
        timeline.push(&update).unwrap();
        assert_eq!((timeline.len(), timeline.cursor()), (2, 1));
        let step = timeline.current().unwrap();
        assert_eq!(step.report, after);
        assert_eq!(step.touched.keys().collect::<Vec<_>>(), [&changed]);
        let diff = step.touched[&changed].as_ref().unwrap();
        assert!((diff.relative_change + 0.5).abs() < 1e-12);

        // Scrubbed back: new patches no longer move the cursor.
        timeline.seek(0);
        assert!(!timeline.is_live());
        timeline
            .push(&GraphPatch::between(&after, &before))
            .unwrap();
        assert_eq!((timeline.len(), timeline.cursor()), (3, 0));
        timeline.seek(99);
        assert!(timeline.is_live());
        assert_eq!(timeline.current().unwrap().report, before);
    }

    #[test]
    fn the_reader_sends_every_line_and_reports_bad_ones() {
        let report = snapshot();
        let mut text = GraphPatch::snapshot(&report).to_ndjson().unwrap();
        text.push_str("\nnot json\n");
        // The last line has no newline: a finished (not followed) stream still reads it.
        text.push_str(
            GraphPatch::between(&report, &report)
                .to_ndjson()
                .unwrap()
                .trim_end(),
        );
        let rx = spawn_reader(Box::new(Cursor::new(text)), false, || {});
        let got: Vec<_> = rx.iter().collect();
        assert_eq!(got.len(), 3);
        assert!(matches!(got[0], Ok(GraphPatch::Snapshot { .. })));
        assert!(got[1].as_ref().unwrap_err().starts_with("ligne NDJSON"));
        assert!(matches!(got[2], Ok(GraphPatch::Update { .. })));
    }

    #[test]
    fn a_followed_file_delivers_lines_appended_later() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("surgeon-gui-follow-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("graph.ndjson");
        let line = GraphPatch::snapshot(&snapshot()).to_ndjson().unwrap();
        std::fs::write(&path, "").unwrap();
        let Input::Stream { reader, follow } = open(&path).unwrap() else {
            panic!("an .ndjson path is a stream");
        };
        assert!(follow);
        let rx = spawn_reader(reader, follow, || {});
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        // Written in two parts: the reader must wait for the end of the line.
        let (head, tail) = line.split_at(line.len() / 2);
        file.write_all(head.as_bytes()).unwrap();
        file.flush().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        file.write_all(tail.as_bytes()).unwrap();
        let got = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(got, Ok(GraphPatch::Snapshot { .. })));
        drop(file);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
