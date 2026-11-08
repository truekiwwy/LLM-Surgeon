//! Terminal identity of `weight-surgeon`: palette, logo, badges, tables, progress bars.
//!
//! Palette (strict): royal blue `#4169E1` (accent: logo blocks, badges, separators, filled
//! bar), bold pure white `#FFFFFF` (values, tensor names, logo text), matte gray `#888888`
//! (paths, durations, keys, empty bar). No background color is ever emitted; states are
//! carried by text (`PASS`, `LOOP`, `FAIL`, `✓ ✗ !`), never by an extra color.
//!
//! JSON safety is structural: in [`ColorMode::Off`] every helper returns an empty string
//! and [`print_banner`] writes nothing, so a `--json` run cannot leak a decorative byte
//! even if a call site forgets to check the mode.
//!
//! Output audit (graphify AST + grep over every write site outside tests):
//!
//! ```text
//! main -> parse_cli_with (help: print_banner + clap render, usage error: emit)
//!      -> run_with -> dispatch -> {check, inspect, apply, eval} -> ui helpers -> Outcome.human
//!                  -> apply -> ObservedBackend -> ui helpers -> out (lines) / err (TTY bar)
//!                  -> emit -> json ? serde_json envelope : Outcome.human + error line
//! json: run_with builds Ui::off() for both streams, emit never writes Outcome.human, and
//!       ObservedBackend's writers go through Ui::off(): no helper output reaches a stream.
//! Writes that bypass `out`/`err`: main's final flush; app::install_panic_hook (only for a
//! panic outside run_with); real_deps (only if the Ctrl-C handler cannot be installed);
//! patch::BackupGuard::drop (only when a guard is dropped uncommitted, i.e. a panic).
//! None of them emits ANSI or a decoration.
//! ```
//!
//! Color mode resolution, per stream (mirrors `anstream::AutoStream::choice`, plus depth):
//!
//! ```text
//! --json                       -> Off        (both streams, wins over everything)
//! NO_COLOR non-empty           -> Plain
//! CLICOLOR_FORCE non-empty, !0 -> color      (even when piped)
//! CLICOLOR=0                   -> Plain
//! not a TTY                    -> Plain
//! TERM=dumb                    -> Plain
//! color: COLORTERM=truecolor|24bit -> TrueColor, otherwise Ansi256
//! ```
//!
//! `anstream` 1.0 does not downgrade RGB to 256 colors (it passes through, strips, or
//! translates to wincon), so the depth is chosen here: `#4169E1` maps to xterm index 62.

use std::io::{self, Write};
use std::path::Path;

use anstyle::{Ansi256Color, Color, RgbColor, Style};
use unicode_width::UnicodeWidthStr;

pub const BLUE: (u8, u8, u8) = (65, 105, 225);
pub const WHITE: (u8, u8, u8) = (255, 255, 255);
pub const GRAY: (u8, u8, u8) = (136, 136, 136);

/// Replace these two constants to change the logo; widths must match per line.
pub const LOGO_BLOCK: [&str; 2] = ["█▀█ █ █ ▀█▀", "█ █ ▀▄▀  █ "];
pub const LOGO_TEXT: [&str; 2] = ["W E I G H T", "S U R G E O N"];

/// Visible width of the widest badge, `[ROLLBACK]`: every badge is padded to it.
const BADGE_WIDTH: usize = 10;
const KEY_WIDTH: usize = 14;
const MAX_WIDTH: usize = 60;
const COLUMN_GAP: &str = "   ";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// `--json`: nothing at all.
    Off,
    /// Text and Unicode decorations, no ANSI.
    Plain,
    Ansi256,
    TrueColor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiConfig {
    pub out: ColorMode,
    pub err: ColorMode,
    /// Enables the in-place progress bar on stderr.
    pub err_is_tty: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ui {
    mode: ColorMode,
    width: usize,
}

impl Ui {
    /// Width: `min(60, $COLUMNS)` when `COLUMNS` is set, else 60.
    pub fn new(mode: ColorMode) -> Self {
        let columns = std::env::var("COLUMNS")
            .ok()
            .and_then(|c| c.trim().parse().ok());
        Self {
            mode,
            width: columns.map_or(MAX_WIDTH, |c: usize| c.clamp(10, MAX_WIDTH)),
        }
    }

    pub fn off() -> Self {
        Self::new(ColorMode::Off)
    }

    pub fn is_off(&self) -> bool {
        self.mode == ColorMode::Off
    }

    pub fn mode(&self) -> ColorMode {
        self.mode
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn detect(json: bool) -> UiConfig {
        use std::io::IsTerminal;
        let env = |key: &str| std::env::var(key).ok();
        let err_is_tty = io::stderr().is_terminal();
        UiConfig {
            out: resolve_mode(&env, io::stdout().is_terminal(), json),
            err: resolve_mode(&env, err_is_tty, json),
            err_is_tty: err_is_tty && !json,
        }
    }

    /// Writes `line` and a newline, or nothing at all in `Off` mode. Write errors are
    /// ignored: a closed pipe must never become a panic.
    pub fn line(&self, w: &mut dyn Write, line: &str) {
        if !self.is_off() {
            let _ = writeln!(w, "{line}");
        }
    }
}

/// Pure resolution, see the table in the module docs.
pub fn resolve_mode(env: &dyn Fn(&str) -> Option<String>, is_tty: bool, json: bool) -> ColorMode {
    let set = |key: &str| env(key).filter(|v| !v.is_empty());
    if json {
        return ColorMode::Off;
    }
    if set("NO_COLOR").is_some() {
        return ColorMode::Plain;
    }
    let forced = set("CLICOLOR_FORCE").is_some_and(|v| v != "0");
    let disabled = env("CLICOLOR").is_some_and(|v| v == "0");
    let dumb = env("TERM").is_some_and(|t| t == "dumb");
    if !forced && (disabled || !is_tty || dumb) {
        return ColorMode::Plain;
    }
    match env("COLORTERM").as_deref() {
        Some("truecolor" | "24bit") => ColorMode::TrueColor,
        _ => ColorMode::Ansi256,
    }
}

/// Nearest xterm-256 color (6x6x6 cube or gray ramp) by squared RGB distance.
pub fn rgb_to_ansi256((r, g, b): (u8, u8, u8)) -> u8 {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let nearest = |c: u8| {
        (0..6)
            .min_by_key(|&i| (i32::from(LEVELS[i]) - i32::from(c)).abs())
            .unwrap_or(0)
    };
    let dist = |(x, y, z): (u8, u8, u8)| {
        let d = |a: u8, b: u8| (i32::from(a) - i32::from(b)).pow(2);
        d(x, r) + d(y, g) + d(z, b)
    };
    let (ri, gi, bi) = (nearest(r), nearest(g), nearest(b));
    let cube = (16 + 36 * ri + 6 * gi + bi) as u8;
    let cube_dist = dist((LEVELS[ri], LEVELS[gi], LEVELS[bi]));
    let gray = (0..24u8)
        .min_by_key(|&i| dist((8 + 10 * i, 8 + 10 * i, 8 + 10 * i)))
        .unwrap_or(0);
    let gray_level = 8 + 10 * gray;
    match dist((gray_level, gray_level, gray_level)) < cube_dist {
        true => 232 + gray,
        false => cube,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tone {
    Accent,
    Strong,
    Dim,
}

fn style(tone: Tone, mode: ColorMode) -> Style {
    let rgb = match tone {
        Tone::Accent => BLUE,
        Tone::Strong => WHITE,
        Tone::Dim => GRAY,
    };
    let color: Color = match mode {
        ColorMode::TrueColor => RgbColor(rgb.0, rgb.1, rgb.2).into(),
        _ => Ansi256Color(rgb_to_ansi256(rgb)).into(),
    };
    let style = Style::new().fg_color(Some(color));
    match tone {
        Tone::Strong => style.bold(),
        _ => style,
    }
}

fn paint(ui: &Ui, tone: Tone, text: &str) -> String {
    match ui.mode {
        ColorMode::Off => String::new(),
        _ if text.is_empty() => String::new(),
        ColorMode::Plain => text.to_owned(),
        mode => {
            let s = style(tone, mode);
            format!("{}{text}{}", s.render(), s.render_reset())
        }
    }
}

/// Blue: brand marks, layer brackets.
pub fn accent(ui: &Ui, text: &str) -> String {
    paint(ui, Tone::Accent, text)
}

/// Bold white: values, metrics, tensor names.
pub fn metric(ui: &Ui, value: impl std::fmt::Display) -> String {
    paint(ui, Tone::Strong, &value.to_string())
}

/// Gray: keys, durations, technical details.
pub fn dim(ui: &Ui, text: &str) -> String {
    paint(ui, Tone::Dim, text)
}

/// Plain text that must vanish in `Off` mode (separators between styled parts).
pub fn text(ui: &Ui, text: &str) -> String {
    match ui.is_off() {
        true => String::new(),
        false => text.to_owned(),
    }
}

fn pad(ui: &Ui, rendered: String, width: usize) -> String {
    if ui.is_off() {
        return rendered;
    }
    let missing = width.saturating_sub(visible_width(&rendered));
    rendered + &" ".repeat(missing)
}

/// `[CHECK]` in blue, padded to the widest badge so the following text lines up.
pub fn badge(ui: &Ui, text: &str) -> String {
    pad(ui, accent(ui, &format!("[{text}]")), BADGE_WIDTH)
}

/// Gray key aligned on a fixed column, already rendered value.
pub fn field(ui: &Ui, key: &str, rendered: &str) -> String {
    if ui.is_off() {
        return String::new();
    }
    format!("  {}{rendered}", pad(ui, dim(ui, key), KEY_WIDTH))
}

/// Gray aligned key, bold white value.
pub fn key_val(ui: &Ui, key: &str, val: &str) -> String {
    field(ui, key, &metric(ui, val))
}

pub fn separator(ui: &Ui) -> String {
    accent(ui, &"─".repeat(ui.width))
}

/// Gray path, truncated in the middle to the UI width.
pub fn path(ui: &Ui, p: &Path) -> String {
    dim(ui, &truncate_middle(&p.display().to_string(), ui.width))
}

/// Keeps both ends of `s` within `max` visible columns, joined by `…`, cut on characters.
pub fn truncate_middle(s: &str, max: usize) -> String {
    if s.width() <= max {
        return s.to_owned();
    }
    let budget = max.saturating_sub(1);
    let char_width = |c: char| unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
    // Two thirds for the end: the file name (a backup's timestamp) matters most.
    let tail_budget = budget - budget / 3;
    let mut tail = Vec::new();
    let mut tail_w = 0;
    for c in s.chars().rev() {
        if tail_w + char_width(c) > tail_budget {
            break;
        }
        tail_w += char_width(c);
        tail.push(c);
    }
    let mut head = String::new();
    let mut head_w = 0;
    for c in s.chars() {
        if head_w + tail_w + char_width(c) > budget {
            break;
        }
        head_w += char_width(c);
        head.push(c);
    }
    head + "…" + &tail.iter().rev().collect::<String>()
}

/// Blue filled part, gray empty part; constant `width`, `done` clamped to `total`.
pub fn progress_bar(ui: &Ui, done: usize, total: usize, width: usize) -> String {
    let filled = match total {
        0 => 0,
        _ => (done.min(total) * width) / total,
    };
    accent(ui, &"█".repeat(filled)) + &dim(ui, &"░".repeat(width - filled))
}

/// Columns aligned on visible width (ANSI stripped, Unicode width), every row padded to
/// the same width. Headers are gray; cells are rendered by the caller.
pub fn table(ui: &Ui, headers: &[&str], rows: &[Vec<String>]) -> String {
    if ui.is_off() {
        return String::new();
    }
    let header_cells: Vec<String> = headers.iter().map(|h| dim(ui, h)).collect();
    let mut widths: Vec<usize> = header_cells.iter().map(|h| visible_width(h)).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if let Some(w) = widths.get_mut(i) {
                *w = (*w).max(visible_width(cell));
            }
        }
    }
    let render = |cells: &[String]| {
        let line: Vec<String> = widths
            .iter()
            .enumerate()
            .map(|(i, &w)| pad(ui, cells.get(i).cloned().unwrap_or_default(), w))
            .collect();
        line.join(COLUMN_GAP) + "\n"
    };
    std::iter::once(render(&header_cells))
        .chain(rows.iter().map(|r| render(r)))
        .collect()
}

/// Removes CSI sequences (`ESC [ ... final byte`).
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if ('\x40'..='\x7e').contains(&c) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

pub fn visible_width(s: &str) -> usize {
    strip_ansi(s).width()
}

fn banner(ui: &Ui) -> String {
    LOGO_BLOCK
        .iter()
        .zip(LOGO_TEXT)
        .map(|(block, txt)| format!("{}  {}\n", accent(ui, block), metric(ui, txt)))
        .collect()
}

/// The two logo lines; writes nothing at all in `Off` mode.
pub fn print_banner(ui: &Ui, out: &mut dyn Write) -> io::Result<()> {
    if ui.is_off() {
        return Ok(());
    }
    out.write_all(banner(ui).as_bytes())
}

/// Help styles: blue headers and usage, bold white literals, gray placeholders. clap only
/// emits them when its own stream detection allows color.
pub fn clap_styles() -> clap::builder::Styles {
    let truecolor = std::env::var("COLORTERM").is_ok_and(|v| v == "truecolor" || v == "24bit");
    clap_styles_for(match truecolor {
        true => ColorMode::TrueColor,
        false => ColorMode::Ansi256,
    })
}

fn clap_styles_for(mode: ColorMode) -> clap::builder::Styles {
    clap::builder::Styles::styled()
        .header(style(Tone::Accent, mode).bold())
        .usage(style(Tone::Accent, mode).bold())
        .literal(style(Tone::Strong, mode))
        .placeholder(style(Tone::Dim, mode))
        .valid(style(Tone::Strong, mode))
        .invalid(style(Tone::Strong, mode))
        .error(style(Tone::Accent, mode).bold())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_modes() -> [Ui; 4] {
        [
            ColorMode::Off,
            ColorMode::Plain,
            ColorMode::Ansi256,
            ColorMode::TrueColor,
        ]
        .map(Ui::new)
    }

    /// Every helper, rendered with one UI.
    fn samples(ui: &Ui) -> Vec<String> {
        let mut banner_bytes = Vec::new();
        print_banner(ui, &mut banner_bytes).unwrap();
        vec![
            String::from_utf8(banner_bytes).unwrap(),
            badge(ui, "CHECK"),
            key_val(ui, "modèle", "14.96 GiB"),
            field(
                ui,
                "chemin",
                &path(ui, Path::new("/très/long/chemin/vers/le/modèle")),
            ),
            separator(ui),
            accent(ui, "[L12]"),
            metric(ui, 42),
            dim(ui, "1.8 s"),
            text(ui, " · "),
            progress_bar(ui, 3, 10, 20),
            table(ui, &["A", "BÉTA"], &[vec![metric(ui, "×"), dim(ui, "→ é")]]),
        ]
    }

    #[test]
    fn banner_is_the_exact_logo_and_aligned() {
        let mut out = Vec::new();
        print_banner(&Ui::new(ColorMode::TrueColor), &mut out).unwrap();
        let colored = String::from_utf8(out).unwrap();
        assert_eq!(
            strip_ansi(&colored),
            "█▀█ █ █ ▀█▀  W E I G H T\n█ █ ▀▄▀  █   S U R G E O N\n"
        );
        let lines: Vec<Vec<char>> = strip_ansi(&colored)
            .lines()
            .map(|l| l.chars().collect())
            .collect();
        assert_eq!(lines[0][9], '█', "center of ▀█▀");
        assert_eq!(lines[1][9], '█', "second line's last block sits under it");
        assert_eq!(LOGO_BLOCK[0].width(), LOGO_BLOCK[1].width());
        assert_eq!(
            lines[0].iter().position(|&c| c == 'W'),
            lines[1].iter().position(|&c| c == 'S')
        );

        let blue = colored.find("38;2;65;105;225").unwrap();
        let white = colored.find("38;2;255;255;255").unwrap();
        let bold = colored.find("\x1b[1m").unwrap();
        assert!(
            blue < white && blue < bold,
            "blocks first, then bold white text"
        );
        assert!(colored[blue..].find('█').unwrap() < colored[blue..].find('W').unwrap());
    }

    #[test]
    fn off_mode_emits_nothing_and_plain_is_stripped_truecolor() {
        for s in samples(&Ui::off()) {
            assert_eq!(s, "");
        }
        let plain = samples(&Ui::new(ColorMode::Plain));
        let ansi = samples(&Ui::new(ColorMode::Ansi256));
        let truecolor = samples(&Ui::new(ColorMode::TrueColor));
        for ((p, a), t) in plain.iter().zip(&ansi).zip(&truecolor) {
            assert!(!p.contains('\x1b'), "{p:?}");
            assert_eq!(*p, strip_ansi(t));
            assert_eq!(*p, strip_ansi(a));
        }
        let joined = ansi.concat();
        assert!(joined.contains("38;5;") && !joined.contains("38;2;"));
        assert!(truecolor.concat().contains("38;2;136;136;136"));
    }

    #[test]
    fn ansi256_equivalents_are_computed() {
        assert_eq!(rgb_to_ansi256(BLUE), 62, "#4169E1 -> #5f5fd7");
        assert_eq!(rgb_to_ansi256(WHITE), 231);
        assert_eq!(rgb_to_ansi256(GRAY), 102, "#878787 beats gray ramp #8a8a8a");
        assert_eq!(rgb_to_ansi256((0, 0, 0)), 16);
        assert_eq!(rgb_to_ansi256((18, 18, 18)), 233);
    }

    #[test]
    fn widths_ignore_ansi_and_count_unicode_columns() {
        let ui = Ui::new(ColorMode::TrueColor);
        assert_eq!(visible_width(&metric(&ui, "modèle à")), 8);
        assert_eq!(visible_width("4096 × 11008 → …"), 16);
        assert_eq!(strip_ansi("\x1b[1m\x1b[38;5;62mé\x1b[0m"), "é");
        assert_eq!(visible_width(&badge(&ui, "VERDICT")), BADGE_WIDTH);

        let t = table(
            &ui,
            &["COUCHE", "PROJ", "TAILLE"],
            &[
                vec![
                    accent(&ui, "[L1]"),
                    metric(&ui, "down_proj"),
                    metric(&ui, "86.0 MiB"),
                ],
                vec![accent(&ui, "[L12]"), metric(&ui, "gâté"), dim(&ui, "—")],
            ],
        );
        let widths: Vec<usize> = t.lines().map(visible_width).collect();
        assert_eq!(widths.len(), 3);
        assert!(
            widths.iter().all(|&w| w == widths[0]),
            "{widths:?}\n{}",
            strip_ansi(&t)
        );
    }

    #[test]
    fn long_paths_are_cut_in_the_middle() {
        let long =
            "/données/modèles/".to_owned() + &"é".repeat(80) + "/model-00001-of-00004.safetensors";
        let cut = truncate_middle(&long, 40);
        assert!(cut.width() <= 40 && cut.contains('…'), "{cut}");
        assert!(
            cut.starts_with("/données") && cut.ends_with(".safetensors"),
            "{cut}"
        );
        assert_eq!(
            truncate_middle("/court/m.safetensors", 40),
            "/court/m.safetensors"
        );
        assert_eq!(truncate_middle("日本語のパスです", 7).width(), 7);
        assert_eq!(truncate_middle("abc", 0), "…");
        let backup = truncate_middle(
            r"D:\data\models\AppData\Local\Temp\claude\scratchpad\visual\work.safetensors.bak-1794157219",
            60,
        );
        assert!(backup.width() <= 60, "{backup}");
        assert!(
            backup.ends_with(r"\work.safetensors.bak-1794157219"),
            "{backup}"
        );
    }

    #[test]
    fn progress_bar_never_panics_and_keeps_its_width() {
        let ui = Ui::new(ColorMode::Plain);
        for (done, total) in [(0, 0), (0, 7), (7, 7), (9, 7), (3, 7)] {
            let bar = progress_bar(&ui, done, total, 20);
            assert_eq!(bar.chars().count(), 20, "{done}/{total}");
        }
        assert_eq!(progress_bar(&ui, 9, 7, 4), "████");
        assert_eq!(progress_bar(&ui, 0, 0, 4), "░░░░");
        assert_eq!(progress_bar(&ui, 1, 2, 4), "██░░");
    }

    #[test]
    fn mode_resolution_truth_table() {
        use ColorMode::*;
        /// (environment, is_tty, json, expected)
        type Case = (
            &'static [(&'static str, &'static str)],
            bool,
            bool,
            ColorMode,
        );
        let cases: &[Case] = &[
            (
                &[("CLICOLOR_FORCE", "1"), ("COLORTERM", "truecolor")],
                true,
                true,
                Off,
            ),
            (&[], true, false, Ansi256),
            (&[("COLORTERM", "truecolor")], true, false, TrueColor),
            (&[("COLORTERM", "24bit")], true, false, TrueColor),
            (&[("COLORTERM", "truecolor")], false, false, Plain),
            (&[("NO_COLOR", "1")], true, false, Plain),
            (&[("NO_COLOR", "")], true, false, Ansi256),
            (
                &[("NO_COLOR", "1"), ("CLICOLOR_FORCE", "1")],
                true,
                false,
                Plain,
            ),
            (&[("CLICOLOR_FORCE", "1")], false, false, Ansi256),
            (
                &[("CLICOLOR_FORCE", "1"), ("COLORTERM", "truecolor")],
                false,
                false,
                TrueColor,
            ),
            (&[("CLICOLOR_FORCE", "0")], false, false, Plain),
            (&[("TERM", "dumb")], true, false, Plain),
            (
                &[("TERM", "dumb"), ("CLICOLOR_FORCE", "1")],
                true,
                false,
                Ansi256,
            ),
            (&[("CLICOLOR", "0")], true, false, Plain),
            (&[("TERM", "xterm-256color")], true, false, Ansi256),
        ];
        for (vars, tty, json, expected) in cases {
            let env = |k: &str| {
                vars.iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            };
            assert_eq!(
                resolve_mode(&env, *tty, *json),
                *expected,
                "{vars:?} tty={tty} json={json}"
            );
        }
    }

    #[test]
    fn clap_styles_follow_the_palette() {
        let styles = clap_styles_for(ColorMode::TrueColor);
        let header = styles.get_header().render().to_string();
        assert!(header.contains("38;2;65;105;225"), "{header:?}");
        assert!(styles
            .get_placeholder()
            .render()
            .to_string()
            .contains("38;2;136;136;136"));
        assert!(clap_styles_for(ColorMode::Ansi256)
            .get_literal()
            .render()
            .to_string()
            .contains("38;5;231"));
        for ui in all_modes() {
            assert_eq!(ui.is_off(), ui.mode() == ColorMode::Off);
        }
    }
}
