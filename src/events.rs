//! The event protocol of long commands: versioned NDJSON lines (`--events <FILE|->`), the
//! same stream the studio (`crates/surgeon-gui`) records in `runs/<id>/events.ndjson` and
//! replays. Every line is derived from the [`ProgressEvent`]s the cores already send, so
//! the CLI, the MCP server and the studio report the same thing. docs/STUDIO.md.
//!
//! Additive only: without `--events` nothing here runs and every output is unchanged.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app::{Outcome, ProgressEvent, ProgressSink, RollbackOutcome};

/// Version of the line format. A reader refuses a newer one; fields added within a version
/// are optional, so an older reader of the same version ignores them.
pub const EVENTS_VERSION: u32 = 1;

/// One NDJSON line: `{"v":1,"seq":0,"t_ms":0,"event":"job_started",...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EventLine {
    pub v: u32,
    /// 0, 1, 2... within one stream: a gap means a lost line.
    pub seq: u64,
    /// Milliseconds since the stream started.
    pub t_ms: u64,
    #[serde(flatten)]
    pub event: SurgeonEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SurgeonEvent {
    /// First line of a stream: the command name of the JSON envelope (`apply`, `bench eval`...).
    JobStarted {
        command: String,
    },
    /// `done` units out of `total` (absent when the producer does not know it).
    Progress {
        done: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total: Option<u64>,
        unit: String,
    },
    /// A node of the weight graph: a tensor name or a graph node id.
    NodeState {
        node_id: String,
        state: NodeStatus,
    },
    Metric {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
        name: String,
        value: f64,
    },
    Log {
        level: LogLevel,
        message: String,
    },
    /// A file the command wrote. `sha256` is absent when the producer did not hash it.
    Artifact {
        kind: String,
        path: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
    },
    Warning {
        message: String,
    },
    /// Last line of a successful command.
    Finished {
        exit_code: i32,
    },
    /// Last line of a failed command; `kind` is the envelope's `error.kind` (`error` if none).
    Failed {
        kind: String,
        message: String,
        exit_code: i32,
    },
    /// Last line of a command stopped by its interruption flag (Ctrl-C, the studio's cancel).
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Pending,
    Running,
    Done,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

impl SurgeonEvent {
    /// `Finished`, `Failed` and `Cancelled` close a stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Finished { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }

    fn info(message: String) -> Self {
        Self::Log {
            level: LogLevel::Info,
            message,
        }
    }

    fn progress(done: usize, total: Option<usize>, unit: &str) -> Self {
        Self::Progress {
            done: done as u64,
            total: total.map(|t| t as u64),
            unit: unit.into(),
        }
    }

    fn metric(node_id: Option<&str>, name: &str, value: f64) -> Option<Self> {
        // NaN and infinities have no JSON form: such a metric is dropped, not written as null.
        value.is_finite().then(|| Self::Metric {
            node_id: node_id.map(Into::into),
            name: name.into(),
            value,
        })
    }

    fn done(node_id: &str) -> Self {
        Self::NodeState {
            node_id: node_id.into(),
            state: NodeStatus::Done,
        }
    }
}

/// The protocol events of one core progress event (none for the purely cosmetic ones).
pub fn from_progress(event: &ProgressEvent) -> Vec<SurgeonEvent> {
    use ProgressEvent as P;
    use SurgeonEvent as S;
    let one = |e: S| vec![e];
    match event {
        P::ApplyStart { patch, operations } => one(S::info(format!(
            "patch {} : {operations} opération(s)",
            patch.display()
        ))),
        P::Warn(message) => one(S::Warning {
            message: message.clone(),
        }),
        P::DryRun { op_id, tensors, .. } => one(S::info(format!(
            "dry-run {op_id} : {} tenseur(s) visé(s)",
            tensors.len()
        ))),
        P::Backup { path } => one(S::Artifact {
            kind: "backup".into(),
            path: path.clone(),
            sha256: None,
        }),
        P::Shift {
            tensor,
            overall_done,
            overall_total,
            ..
        } => vec![
            S::done(tensor),
            S::progress(*overall_done, Some(*overall_total), "tensor"),
        ],
        P::OpDone {
            op_id, max_delta, ..
        }
        | P::ProjectDone {
            op_id, max_delta, ..
        } => S::metric(Some(op_id), "max_delta", f64::from(*max_delta))
            .into_iter()
            .collect(),
        P::Commit { seconds } => one(S::info(format!("écriture validée en {seconds:.1} s"))),
        P::Rollback(outcome) => one(S::Warning {
            message: match outcome {
                RollbackOutcome::Restored => "rollback : modèle restauré".into(),
                RollbackOutcome::Failed { backups } => format!(
                    "rollback échoué : {} sauvegarde(s) à restaurer à la main",
                    backups.len()
                ),
                RollbackOutcome::NoBackup => {
                    "rollback impossible : aucune sauvegarde, modèle partiellement modifié".into()
                }
            },
        }),
        P::EvalPrompt { id, done, total } => {
            vec![S::done(id), S::progress(*done, Some(*total), "prompt")]
        }
        P::ExtractStart {
            model, layer, site, ..
        } => one(S::info(format!(
            "extraction {} : couche {layer}, site {site}",
            model.display()
        ))),
        P::ExtractLoading { done, total } => one(S::progress(*done, Some(*total), "weight_file")),
        P::ExtractPrompt { set, done, total } => one(S::progress(
            *done,
            Some(*total),
            &format!("prompt_{}", set.as_str()),
        )),
        P::ExtractWriting => one(S::info("écriture du vecteur".into())),
        P::BenchDumpPrompt { done, total } => one(S::progress(*done, Some(*total), "prompt")),
        P::BenchLogitsBlock { done, total } => one(S::progress(*done, Some(*total), "block")),
        P::BenchPplChunk { done, total } => one(S::progress(*done, *total, "chunk")),
        P::BenchEvalQuestion { done, total } => one(S::progress(*done, Some(*total), "question")),
        P::BenchRefusalPrompt { done, total } => one(S::progress(*done, Some(*total), "prompt")),
        P::BenchJudgeVerdict { done, total } => one(S::progress(*done, Some(*total), "verdict")),
        P::AutotuneForward { done, total } => one(S::progress(*done, Some(*total), "forward")),
        P::AutotuneSolve => one(S::info("régression ridge".into())),
        P::GraphStart {
            model, matrices, ..
        } => one(S::info(format!(
            "graphe {} : {matrices} matrice(s)",
            model.display()
        ))),
        P::GraphNode { id, done, total } => {
            vec![S::done(id), S::progress(*done, Some(*total), "matrix")]
        }
        P::GraphEdges { nodes, edges } => [
            S::metric(None, "nodes", *nodes as f64),
            S::metric(None, "edges", *edges as f64),
        ]
        .into_iter()
        .flatten()
        .collect(),
        P::GraphWrite { files } => one(S::info(format!("{files} fichier(s) écrit(s)"))),
        P::GraphDone { seconds, .. } => S::metric(None, "seconds", *seconds).into_iter().collect(),
        P::Surgery(ev) => match crate::surgery::events::counter(ev) {
            Some((unit, name, done, total)) => {
                vec![S::done(name), S::progress(done, Some(total), unit)]
            }
            None => crate::surgery::events::render(&crate::ui::Ui::off(), ev)
                .map(S::info)
                .into_iter()
                .collect(),
        },
    }
}

/// The closing line of a command: `Cancelled` when it failed after its interruption flag
/// was raised, else `Finished` or `Failed`.
pub fn terminal(outcome: &Outcome, interrupted: bool) -> SurgeonEvent {
    if outcome.is_ok() {
        return SurgeonEvent::Finished {
            exit_code: outcome.exit_code,
        };
    }
    if interrupted {
        return SurgeonEvent::Cancelled;
    }
    let error = outcome.error.as_deref();
    SurgeonEvent::Failed {
        kind: error
            .and_then(|e| e.kind.clone())
            .unwrap_or_else(|| "error".into()),
        message: error.map(|e| e.message.clone()).unwrap_or_default(),
        exit_code: outcome.exit_code,
    }
}

/// Writes numbered lines, one flush per line: a crash loses at most the line being written.
pub struct EventWriter {
    out: Box<dyn Write + Send>,
    seq: u64,
    start: Instant,
    error: Option<io::Error>,
}

impl EventWriter {
    pub fn new(out: Box<dyn Write + Send>) -> Self {
        Self {
            out,
            seq: 0,
            start: Instant::now(),
            error: None,
        }
    }

    /// `-` is the standard error (stdout stays the command's output, envelope included);
    /// anything else is a file, created or truncated.
    pub fn open(target: &Path) -> io::Result<Self> {
        let out: Box<dyn Write + Send> = match target.as_os_str() == "-" {
            true => Box::new(io::stderr()),
            false => Box::new(BufWriter::new(File::create(target)?)),
        };
        Ok(Self::new(out))
    }

    /// Writes the line and hands it back (the studio forwards it to its window).
    pub fn emit(&mut self, event: SurgeonEvent) -> EventLine {
        let line = EventLine {
            v: EVENTS_VERSION,
            seq: self.seq,
            t_ms: self.start.elapsed().as_millis() as u64,
            event,
        };
        self.seq += 1;
        let written = serde_json::to_writer(&mut self.out, &line)
            .map_err(io::Error::from)
            .and_then(|()| self.out.write_all(b"\n"))
            .and_then(|()| self.out.flush());
        if let (Err(e), None) = (written, &self.error) {
            self.error = Some(e);
        }
        line
    }

    /// The first write error, if any: the command still ran, its stream is incomplete.
    pub fn take_error(&mut self) -> Option<io::Error> {
        self.error.take()
    }
}

impl ProgressSink for EventWriter {
    fn event(&mut self, event: ProgressEvent) {
        for e in from_progress(&event) {
            self.emit(e);
        }
    }
}

/// Parses one line; refuses a newer protocol version.
pub fn parse_line(line: &str) -> Result<EventLine, String> {
    let parsed: EventLine = serde_json::from_str(line).map_err(|e| e.to_string())?;
    if parsed.v > EVENTS_VERSION {
        return Err(format!(
            "protocole d'événements v{} : cette version lit jusqu'à v{EVENTS_VERSION}",
            parsed.v
        ));
    }
    Ok(parsed)
}

/// JSON Schema (draft-07) of one line: `surgeon schema --kind events`.
pub fn schema() -> Value {
    let generator = SchemaSettings::draft07().into_generator();
    serde_json::to_value(generator.into_root_schema_for::<EventLine>()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_newer_version_is_refused_and_unknown_fields_are_ignored() {
        let ok = r#"{"v":1,"seq":3,"t_ms":5,"event":"cancelled","later_field":true}"#;
        assert_eq!(parse_line(ok).unwrap().event, SurgeonEvent::Cancelled);
        let newer = r#"{"v":2,"seq":0,"t_ms":0,"event":"cancelled"}"#;
        assert!(parse_line(newer).unwrap_err().contains("v2"));
    }

    #[test]
    fn a_non_finite_metric_is_dropped() {
        let ev = ProgressEvent::OpDone {
            op_id: "a".into(),
            max_delta: f32::NAN,
            tensors: 1,
        };
        assert!(from_progress(&ev).is_empty());
    }

    #[test]
    fn lines_are_numbered_from_zero() {
        let mut w = EventWriter::new(Box::new(Vec::new()));
        w.emit(SurgeonEvent::Cancelled);
        w.emit(SurgeonEvent::Cancelled);
        assert_eq!(w.seq, 2);
        assert!(w.take_error().is_none());
    }
}
