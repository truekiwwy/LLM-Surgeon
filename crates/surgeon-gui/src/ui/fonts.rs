//! The embedded typefaces (`assets/fonts`, SIL OFL 1.1, licenses next to the files):
//! Barlow for navigation, titles and running text, JetBrains Mono for tensors, shapes,
//! norms, hashes and logs. Nothing is read from the system at run time.

use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};

use super::theme;

const SANS: &[u8] = include_bytes!("../../assets/fonts/Barlow-Regular.ttf");
const SANS_MEDIUM: &[u8] = include_bytes!("../../assets/fonts/Barlow-Medium.ttf");
const MONO: &[u8] = include_bytes!("../../assets/fonts/JetBrainsMono-Regular.ttf");

/// Symbols the interface draws as text. Barlow lacks some: every family falls back to the
/// monospace face, which has them all (`tests/ui_type.rs`).
pub const SYMBOLS: &str = "‖Δ×→←σκ·…≥≤±—°";
/// The accented letters of the French locale.
pub const FRENCH: &str = "àâäçéèêëîïôöùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŒÆ«»’";

pub fn definitions() -> FontDefinitions {
    let mut fonts = FontDefinitions::empty();
    for (name, bytes) in [("sans", SANS), ("sans-medium", SANS_MEDIUM), ("mono", MONO)] {
        fonts
            .font_data
            .insert(name.into(), Arc::new(FontData::from_static(bytes)));
    }
    let chain = |first: &str, second: &str| vec![first.to_owned(), second.to_owned()];
    fonts
        .families
        .insert(FontFamily::Proportional, chain("sans", "mono"));
    fonts
        .families
        .insert(FontFamily::Monospace, chain("mono", "sans"));
    fonts
        .families
        .insert(theme::medium(), chain("sans-medium", "mono"));
    fonts
}

pub fn install(ctx: &egui::Context) {
    ctx.set_fonts(definitions());
}
