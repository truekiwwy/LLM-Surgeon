//! surgeon-gui: native viewer of the weight graph of `surgeon graph`, in pure Rust (egui).
//!
//! It only draws: the analysis, the two 3D layouts and the patch format all come from the
//! `weight_surgeon` crate (`graph::layout`, `graph::stream`), and the backend settings are
//! the user `config.toml` the CLI reads (`backend::config`).
//!
//! Layout of the crate:
//! - [`source`]: `graph.json` / NDJSON input, the stream reader thread, the replay history.
//! - [`camera`]: orbit camera, projection and picking.
//! - [`viewport`]: the 3D canvas.
//! - [`backend_dialog`]: the backend setup modal.
//! - [`app`]: state and the docked panels.
//! - [`theme`]: colors and style.
#![deny(unsafe_code)]

pub mod app;
pub mod backend_dialog;
pub mod camera;
pub mod source;
pub mod theme;
pub mod viewport;
