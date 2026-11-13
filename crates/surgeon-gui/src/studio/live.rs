//! The 3D layer of a run. Its events, live from a job or read back from `events.ndjson`,
//! fold into a [`RunView`] (the same fold either way, so a replay ends on what was shown
//! live); [`Layer`] maps that view onto the nodes of the graph on screen, by graph id or
//! tensor name, and [`Replay`] moves a time cursor over a recorded run.

use std::path::{Path, PathBuf};
use std::time::Duration;

use eframe::egui::Color32;
use weight_surgeon::events::{EventLine, NodeStatus};
use weight_surgeon::graph::output::ColorScale;

use super::bus::{self, RunView};
use crate::source::Step;

/// What a run does to the graph, from its `job_started` command (`surgeon_apply` in the
/// studio, `apply` or `bench eval` from `surgeon --events`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JobKind {
    Graph,
    Extract,
    Apply,
    Bench,
    Pipeline,
    #[default]
    Other,
}

impl JobKind {
    pub fn of(command: &str) -> Self {
        let head = command
            .trim_start_matches("surgeon_")
            .split([' ', '_'])
            .next()
            .unwrap_or("");
        match head {
            "graph" => Self::Graph,
            "extract" => Self::Extract,
            "apply" => Self::Apply,
            "bench" | "eval" => Self::Bench,
            "pipeline" => Self::Pipeline,
            _ => Self::Other,
        }
    }

    /// The per-node metric the layer colors by.
    pub fn metric(self) -> Option<&'static str> {
        match self {
            Self::Apply => Some("max_abs_delta"),
            _ => None,
        }
    }
}

/// A run mapped onto one graph state. Refilled in place every frame: the vectors keep
/// their capacity.
#[derive(Debug, Default)]
pub struct Layer {
    pub kind: JobKind,
    /// Per node index of the step.
    pub states: Vec<Option<NodeStatus>>,
    /// The kind's metric per node index.
    pub values: Vec<Option<f64>>,
    /// Its color: the report's own scale, on the interface's data steps.
    pub colors: Vec<Option<Color32>>,
    /// Node ids of the run that are not in this graph (prompts, operations, another model).
    pub unmatched: usize,
    /// The node the run reached last.
    pub last: Option<usize>,
    /// `[min, max]` of the metric.
    pub range: Option<(f64, f64)>,
    sorted: Vec<f64>,
}

impl Layer {
    pub fn fill(&mut self, view: &RunView, step: &Step) {
        let n = step.report.nodes.len();
        self.kind = JobKind::of(view.command.as_deref().unwrap_or(""));
        self.states.clear();
        self.states.resize(n, None);
        self.values.clear();
        self.values.resize(n, None);
        self.colors.clear();
        self.colors.resize(n, None);
        self.unmatched = 0;
        for (id, state) in &view.nodes {
            match step.index.get(id) {
                Some(&i) => self.states[i] = Some(*state),
                None => self.unmatched += 1,
            }
        }
        self.last = view
            .last_node
            .as_ref()
            .and_then(|id| step.index.get(id))
            .copied();
        if let Some(metric) = self.kind.metric() {
            for ((node, name), value) in &view.metrics {
                let index = node.as_ref().and_then(|id| step.index.get(id));
                if let (true, Some(&i)) = (name == metric, index) {
                    self.values[i] = Some(*value);
                }
            }
        }
        self.sorted.clear();
        self.sorted.extend(self.values.iter().flatten());
        self.sorted.sort_by(f64::total_cmp);
        self.range = self
            .sorted
            .first()
            .zip(self.sorted.last())
            .map(|(a, b)| (*a, *b));
        let params = &step.report.parameters;
        let scale = ColorScale::from_name(&params.color_scale);
        for (color, value) in self.colors.iter_mut().zip(&self.values) {
            *color = value.map(|v| {
                // The interface's one data scale, whatever palette the report names.
                crate::theme::data_step(scale.position(v, &self.sorted))
            });
        }
    }

    pub fn count(&self, state: NodeStatus) -> usize {
        self.states.iter().filter(|s| **s == Some(state)).count()
    }
}

/// A recorded run with a cursor: the view is the fold of the first `cursor` lines, so any
/// position shows exactly what was on screen when that line arrived.
///
/// shortcut: keeps every line in memory and refolds from the start when the cursor moves
/// back; fine for runs of a few hundred thousand lines, add periodic snapshots past that.
#[derive(Debug, Clone)]
pub struct Replay {
    pub file: PathBuf,
    lines: Vec<EventLine>,
    cursor: usize,
    view: RunView,
    pub playing: bool,
    /// Stream time per wall time.
    pub speed: f64,
    clock_ms: f64,
}

impl Replay {
    /// Opens on the end of the run: the result first, playback on demand.
    pub fn open(file: &Path) -> Result<Self, String> {
        Ok(Self::new(file.to_path_buf(), bus::read_lines(file)?))
    }

    pub fn new(file: PathBuf, lines: Vec<EventLine>) -> Self {
        let mut replay = Self {
            file,
            lines,
            cursor: 0,
            view: RunView::default(),
            playing: false,
            speed: 1.0,
            clock_ms: 0.0,
        };
        replay.seek(usize::MAX);
        replay
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Lines folded so far.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn view(&self) -> &RunView {
        &self.view
    }

    /// Stream time of the last line folded.
    pub fn t_ms(&self) -> u64 {
        self.cursor.checked_sub(1).map_or(0, |i| self.lines[i].t_ms)
    }

    pub fn duration_ms(&self) -> u64 {
        self.lines.last().map_or(0, |l| l.t_ms)
    }

    pub fn seek(&mut self, cursor: usize) {
        let cursor = cursor.min(self.lines.len());
        if cursor < self.cursor {
            self.view = RunView::default();
            self.cursor = 0;
        }
        for line in &self.lines[self.cursor..cursor] {
            self.view.apply(line);
        }
        self.cursor = cursor;
        self.clock_ms = self.t_ms() as f64;
    }

    /// Play or pause; playing from the end starts over.
    pub fn toggle(&mut self) {
        if !self.playing && self.cursor == self.lines.len() {
            self.seek(0);
        }
        self.playing = !self.playing;
    }

    /// Moves the clock by `dt` of wall time while playing and folds the lines it passed.
    /// Returns whether the view changed; stops at the last line.
    pub fn advance(&mut self, dt: Duration) -> bool {
        if !self.playing {
            return false;
        }
        self.clock_ms += dt.as_secs_f64() * 1000.0 * self.speed;
        let start = self.cursor;
        while self
            .lines
            .get(self.cursor)
            .is_some_and(|l| l.t_ms as f64 <= self.clock_ms)
        {
            self.view.apply(&self.lines[self.cursor]);
            self.cursor += 1;
        }
        if self.cursor == self.lines.len() {
            self.playing = false;
        }
        self.cursor != start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Timeline;
    use crate::testkit;
    use weight_surgeon::events::{LogLevel, SurgeonEvent};

    fn line(seq: u64, t_ms: u64, event: SurgeonEvent) -> EventLine {
        EventLine {
            v: 1,
            seq,
            t_ms,
            event,
        }
    }

    fn state(id: &str, state: NodeStatus) -> SurgeonEvent {
        SurgeonEvent::NodeState {
            node_id: id.into(),
            state,
        }
    }

    fn metric(id: &str, value: f64) -> SurgeonEvent {
        SurgeonEvent::Metric {
            node_id: Some(id.into()),
            name: "max_abs_delta".into(),
            value,
        }
    }

    /// An apply over the first nodes of `report`: running, then done with a |ΔW|; the
    /// third one fails.
    fn apply_run(timeline: &Timeline) -> Vec<EventLine> {
        let nodes = &timeline.current().unwrap().report.nodes;
        let mut events = vec![SurgeonEvent::JobStarted {
            command: "surgeon_apply".into(),
        }];
        for (k, node) in nodes.iter().take(3).enumerate() {
            events.push(state(&node.tensor, NodeStatus::Running));
            match k {
                2 => events.push(state(&node.tensor, NodeStatus::Failed)),
                _ => {
                    events.push(state(&node.tensor, NodeStatus::Done));
                    events.push(metric(&node.tensor, 0.01 * (k + 1) as f64));
                }
            }
        }
        events.push(state("prompt-7", NodeStatus::Done));
        events.push(SurgeonEvent::Failed {
            kind: "io".into(),
            message: "disque plein".into(),
            exit_code: 1,
        });
        events
            .into_iter()
            .enumerate()
            .map(|(i, e)| line(i as u64, 100 * i as u64, e))
            .collect()
    }

    fn fold(lines: &[EventLine]) -> RunView {
        let mut view = RunView::default();
        lines.iter().for_each(|l| view.apply(l));
        view
    }

    #[test]
    fn job_kinds_come_from_studio_tools_and_cli_commands() {
        assert_eq!(JobKind::of("surgeon_apply"), JobKind::Apply);
        assert_eq!(JobKind::of("apply"), JobKind::Apply);
        assert_eq!(JobKind::of("bench eval"), JobKind::Bench);
        assert_eq!(JobKind::of("surgeon_eval"), JobKind::Bench);
        assert_eq!(JobKind::of("surgeon_graph"), JobKind::Graph);
        assert_eq!(JobKind::of("pipeline plan"), JobKind::Pipeline);
        assert_eq!(JobKind::of("surgeon_extract"), JobKind::Extract);
        assert_eq!(JobKind::of("studio_copy"), JobKind::Other);
        assert_eq!(JobKind::Apply.metric(), Some("max_abs_delta"));
        assert_eq!(JobKind::Graph.metric(), None);
    }

    #[test]
    fn the_layer_maps_states_and_deltas_onto_graph_nodes() {
        let timeline = Timeline::from_report(testkit::graph(12));
        let step = timeline.current().unwrap();
        let view = fold(&apply_run(&timeline));
        let mut layer = Layer::default();
        layer.fill(&view, step);
        assert_eq!(layer.kind, JobKind::Apply);
        assert_eq!(layer.states.len(), 12);
        assert_eq!(
            layer.states[0],
            Some(NodeStatus::Done),
            "matched by tensor name"
        );
        assert_eq!(layer.states[2], Some(NodeStatus::Failed));
        assert_eq!(layer.states[3], None);
        assert_eq!(layer.unmatched, 1, "the prompt is not a graph node");
        assert_eq!(layer.last, None, "the last node reached is the prompt");
        assert_eq!(layer.range, Some((0.01, 0.02)));
        assert!(layer.colors[0].is_some() && layer.colors[2].is_none());
        assert_ne!(
            layer.colors[0], layer.colors[1],
            "the scale separates the two deltas"
        );
        assert_eq!(
            (
                layer.count(NodeStatus::Done),
                layer.count(NodeStatus::Failed)
            ),
            (2, 1)
        );
        // Graph ids match as well, and refilling reuses the buffers.
        let mut view = RunView::default();
        view.apply(&line(
            0,
            0,
            state(&step.report.nodes[5].id, NodeStatus::Running),
        ));
        let capacity = layer.states.capacity();
        layer.fill(&view, step);
        assert_eq!(layer.states[5], Some(NodeStatus::Running));
        assert_eq!(
            (layer.last, layer.kind, layer.range),
            (Some(5), JobKind::Other, None)
        );
        assert_eq!(layer.states.capacity(), capacity);
    }

    #[test]
    fn duplicates_reordering_and_losses_never_move_a_node_back() {
        let timeline = Timeline::from_report(testkit::graph(4));
        let clean = apply_run(&timeline);
        // Every line twice, a late copy of the first lines, and two lines lost.
        let mut noisy: Vec<EventLine> = clean.iter().flat_map(|l| [l.clone(), l.clone()]).collect();
        noisy.extend(clean[..3].iter().cloned());
        noisy.retain(|l| l.seq != 1);
        let view = fold(&noisy);
        let reference = fold(&clean);
        assert_eq!(view.nodes, reference.nodes);
        assert_eq!(view.end, reference.end);
        assert_eq!(view.gaps, 1, "the lost line is counted");
        // The copies of every line but the lost one, and the late seq 0 and 2.
        assert_eq!(view.stale, clean.len() as u64 - 1 + 2);
    }

    #[test]
    fn the_cursor_shows_the_fold_of_the_lines_before_it_both_ways() {
        let timeline = Timeline::from_report(testkit::graph(4));
        let lines = apply_run(&timeline);
        let mut replay = Replay::new("events.ndjson".into(), lines.clone());
        assert_eq!(replay.cursor(), lines.len(), "opens on the result");
        assert_eq!(replay.view(), &fold(&lines));
        for k in (0..=lines.len()).rev().chain(0..=lines.len()) {
            replay.seek(k);
            assert_eq!(replay.view(), &fold(&lines[..k]), "cursor {k}");
        }
        replay.seek(3);
        assert_eq!(replay.t_ms(), 200);
        assert_eq!(replay.duration_ms(), 100 * (lines.len() as u64 - 1));
    }

    #[test]
    fn playback_follows_stream_time_and_stops_at_the_end() {
        let timeline = Timeline::from_report(testkit::graph(4));
        let lines = apply_run(&timeline);
        let mut replay = Replay::new("events.ndjson".into(), lines.clone());
        assert!(!replay.advance(Duration::from_secs(1)), "paused");
        replay.toggle();
        assert_eq!(replay.cursor(), 0, "playing from the end starts over");
        assert!(replay.advance(Duration::from_millis(150)));
        assert_eq!(replay.cursor(), 2, "lines at 0 and 100 ms");
        replay.speed = 4.0;
        replay.advance(Duration::from_millis(100));
        assert_eq!(replay.cursor(), 6, "550 ms of stream");
        replay.advance(Duration::from_secs(60));
        assert_eq!(replay.cursor(), lines.len());
        assert!(!replay.playing, "stopped at the end");
    }

    #[test]
    fn a_high_rate_run_keeps_bounded_memory() {
        let timeline = Timeline::from_report(testkit::graph(500));
        let step = timeline.current().unwrap();
        let nodes = &step.report.nodes;
        let mut view = RunView::default();
        let mut layer = Layer::default();
        for seq in 0..200_000u64 {
            let node = &nodes[(seq / 4) as usize % nodes.len()].tensor;
            let event = match seq % 4 {
                0 => state(node, NodeStatus::Running),
                1 => state(node, NodeStatus::Done),
                2 => metric(node, seq as f64),
                _ => SurgeonEvent::Log {
                    level: LogLevel::Info,
                    message: format!("ligne {seq}"),
                },
            };
            view.apply(&line(seq, seq, event));
            if seq % 10_000 == 0 {
                layer.fill(&view, step);
            }
        }
        assert_eq!(view.lines, 200_000);
        assert_eq!(view.nodes.len(), 500, "one entry per node");
        assert_eq!(view.metrics.len(), 500, "one entry per (node, metric)");
        assert_eq!(view.logs.len(), bus::MAX_LOGS);
        layer.fill(&view, step);
        assert_eq!(layer.states.len(), 500);
        assert_eq!(
            layer.count(NodeStatus::Running) + layer.count(NodeStatus::Done),
            500
        );
    }

    #[test]
    fn the_v1_fixture_replays() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/events/v1.ndjson");
        let replay = Replay::open(&fixture).unwrap();
        assert_eq!(replay.len(), 13);
        assert_eq!(replay.view().gaps, 0);
        assert_eq!(replay.view().command.as_deref(), Some("apply"));
        assert_eq!(
            replay.view().nodes["model.layers.1.mlp.down_proj.weight"],
            NodeStatus::Running
        );
    }
}
