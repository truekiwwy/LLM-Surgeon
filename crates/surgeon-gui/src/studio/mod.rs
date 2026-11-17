//! The studio foundation (docs/STUDIO.md): persistent projects, a job manager that runs the
//! engine as a library, and one event stream for the 3D view, the log and the agent.
//!
//! - [`project`]: the project folder, its versioned manifest and its states.
//! - [`copy`]: working copies, the only models the studio writes.
//! - [`jobs`]: queue, heavy-job lock, budgets, cancellation, crash recovery.
//! - [`exec`]: the engine executor (`weight_surgeon::mcp::call_tool`).
//! - [`bus`]: the NDJSON recorder, the 60 Hz fold and the replay.
//! - [`ledger`]: the append-only experiment ledger.
//! - [`forms`]: launch forms generated from the engine's tool schemas.
//! - [`estimate`]: cost and time before launch, and the budget check.
//! - [`live`]: the 3D layer of a run, live or replayed with a time cursor.
//! - [`lock`]: the kernel lock that keeps the source model read-only while its project is open.
//! - [`selftest`]: `surgeon-gui --self-test`, all of the above without a window.
//! - [`view`]: the Projects workspace of the window.

pub mod bus;
pub mod copy;
pub mod estimate;
pub mod exec;
pub mod forms;
pub mod jobs;
pub mod ledger;
pub mod live;
pub mod lock;
pub mod project;
pub mod results;
pub mod selftest;
pub mod view;
