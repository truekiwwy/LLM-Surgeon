//! `.surgery` differential patches: low-rank approximation of `tuned - base`, binary
//! container, and in-place application on SafeTensors weights.
//!
//! Module graph (P13, see `docs/SPECIFICATION.md` for the byte layout):
//!
//! ```text
//! cli_args  -> format (FactorDtype)
//! diff      -> rsvd, format, writer, events, error
//! apply     -> reader, lowrank, target, events, error      (try_apply = CLI entry point)
//! lowrank   -> engine::DType, error                        (shared with lora::merge)
//! writer / reader / schema -> format, error
//! rsvd      -> linalg::{qr, jacobi}, error                 (shared with lora::extract)
//! ```

pub mod apply;
pub mod cli_args;
pub mod diff;
pub mod error;
pub mod events;
pub mod format;
pub mod lowrank;
pub mod reader;
pub mod rsvd;
pub mod schema;
pub mod target;
pub mod writer;

pub use apply::try_apply;
pub use diff::core_diff;
pub use error::{Result, SurgeryError};
pub use events::{NullObserver, SinkObserver, SurgeryEvent, SurgeryObserver};
