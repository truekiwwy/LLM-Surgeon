//! surgeon-gui (LLM Surgery Studio): native viewer of the weight graph of `surgeon graph`
//! and surgery workbench, in pure Rust (egui).
//!
//! It only draws and wires: the analysis, the two 3D layouts, the patch format and its
//! validation, and every command a card runs come from the `weight_surgeon` crate
//! (`graph`, `patch`, `app::run_with`); the backend settings are the user `config.toml`
//! the CLI reads (`backend::config`).
//!
//! Layout of the crate (the desktop shell: [`desktop`] screens and wizard, [`actions`] what
//! they do, [`install`] / [`extract`] / [`net`] / [`releases`] / [`sample_model`] the
//! downloads, [`diag`] diagnostics, [`instance`] single instance, [`raster`] off-screen
//! rendering, [`selftest`], [`logging`], [`settings`], [`paths`], [`platform`], [`update`]):
//! - [`source`]: `graph.json` / NDJSON input, the stream reader thread, the replay history.
//! - [`camera`]: orbit camera, projection and picking.
//! - [`viewport`]: the 3D canvas.
//! - [`canvas`]: the studio workspace (cards, wires, run plans) and its infinite canvas.
//! - [`editor`]: the patch.json editor, live validation and 3D target preview.
//! - [`runner`]: background jobs (CLI commands in process, backup restore).
//! - [`backend_dialog`]: the backend setup modal.
//! - [`app`]: state and the docked panels.
//! - [`theme`]: colors and style.
#![deny(unsafe_code)]

pub mod actions;
pub mod app;
pub mod backend_dialog;
pub mod camera;
pub mod canvas;
pub mod cli;
pub mod desktop;
pub mod diag;
pub mod editor;
pub mod extract;
pub mod install;
pub mod instance;
pub mod logging;
pub mod net;
pub mod paths;
pub mod platform;
pub mod raster;
pub mod releases;
pub mod runner;
pub mod sample_model;
pub mod selftest;
pub mod settings;
pub mod source;
pub mod task;
#[cfg(test)]
mod testkit;
pub mod theme;
pub mod update;
pub mod viewport;
