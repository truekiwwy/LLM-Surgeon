//! surgeon-gui (LLM Surgery Studio): native viewer of the weight graph of `surgeon graph`
//! and surgery workbench, in pure Rust (egui).
//!
//! It only draws and wires: the analysis, the two 3D layouts, the patch format and its
//! validation, and every command a card runs come from the `weight_surgeon` crate
//! (`graph`, `patch`, `app::run_with`); the backend settings are the user `config.toml`
//! the CLI reads (`backend::config`).
//!
//! Layout of the crate (the desktop shell: [`desktop`] screens and wizard, [`actions`] what
//! they do, [`install`] / [`extract`] / [`hipfire_setup`] / [`net`] / [`releases`] / [`sample_model`] the
//! downloads, [`diag`] diagnostics, [`instance`] single instance, [`raster`] off-screen
//! rendering, [`selftest`], [`logging`], [`settings`], [`paths`], [`platform`], [`update`]):
//! - [`source`]: `graph.json` / NDJSON input, the stream reader thread, the replay history.
//! - [`camera`]: orbit camera, projection and picking.
//! - [`viewport`]: the 3D canvas.
//! - [`runner`]: background jobs (CLI commands in process, backup restore).
//! - [`backend_choice`]: the default backend from the hardware, Rust engines only.
//! - [`backend_dialog`]: the backend setup modal.
//! - [`hipfire_daemon`]: the hipfire daemon started and stopped with the window.
//! - [`studio`]: projects, the job manager and the event stream (docs/STUDIO.md).
//! - [`app`]: the window state and what the views ask it to do.
//! - [`ui`]: design tokens, components, the shell (title bar, activity bar, palette,
//!   settings dialog) and the two views; [`demo`]: the `--demo-data` workspace.
#![deny(unsafe_code)]

pub mod actions;
pub mod agent;
pub mod app;
pub mod backend_choice;
pub mod backend_dialog;
pub mod camera;
pub mod cli;
pub mod demo;
pub mod desktop;
pub mod diag;
pub mod extract;
pub mod hipfire_daemon;
pub mod hipfire_setup;
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
pub mod scene;
pub mod selftest;
pub mod settings;
pub mod source;
pub mod studio;
pub mod task;
#[cfg(test)]
mod testkit;
pub mod ui;
pub mod update;
pub mod viewport;

/// The design tokens under their historical path.
pub use ui::theme;
