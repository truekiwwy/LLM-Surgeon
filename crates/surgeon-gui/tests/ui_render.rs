//! The real window, rendered off screen on the `--demo-data` workspace: every tab in both
//! languages, the keyboard (tabs, palette, tree), both modes, the theme's contrasts and type
//! coverage, and the accent budget of each screen (docs/design/DIRECTION.md, section 2).

use eframe::egui::{self, Event, Key, Modifiers, Pos2, RawInput, Rect, Vec2};
use surgeon_gui::actions::Env;
use surgeon_gui::agent::policy::AgentMode;
use surgeon_gui::app::{Mode, SurgeonApp, Tab};
use surgeon_gui::desktop::Desktop;
use surgeon_gui::settings::{Lang, Settings};
use surgeon_gui::ui::{components, fonts, theme};
use surgeon_gui::{demo, raster};

const SIZE: [u32; 2] = [1440, 900];

struct Bench {
    ctx: egui::Context,
    app: SurgeonApp,
    _dir: tempfile::TempDir,
}

fn bench(lang: Lang) -> Bench {
    let dir = tempfile::tempdir().unwrap();
    let env = Env {
        components: dir.path().join("components"),
        config: None,
        settings: dir.path().join("settings.json"),
    };
    let settings = Settings {
        language: Some(lang),
        wizard_done: true,
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let desktop = Desktop::detached(None, settings, env);
    let mut app = SurgeonApp::with_shell(&ctx, None, desktop, dir.path().join("projects"));
    demo::install(&ctx, &mut app).unwrap();
    app.dialog.open = false;
    app.set_motion(&ctx, false);
    Bench {
        ctx,
        app,
        _dir: dir,
    }
}

impl Bench {
    fn texts(&mut self, tab: Tab) -> Vec<String> {
        self.app.tab = tab;
        let app = &mut self.app;
        raster::render(&self.ctx, SIZE, 1.0, |ui| app.draw(ui)).texts
    }

    /// One frame with these input events.
    fn press(&mut self, events: Vec<Event>) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 900.0))),
            events,
            ..Default::default()
        };
        let app = &mut self.app;
        let mut out = self.ctx.run_ui(input, |ui| app.draw(ui));
        std::mem::take(&mut out.textures_delta).clear();
    }
}

fn key(key: Key, modifiers: Modifiers) -> Vec<Event> {
    vec![Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }]
}

/// A key of the locales left on screen (`t` returns the key when its text is missing).
fn raw_key(text: &str) -> bool {
    let keys: std::collections::BTreeMap<String, String> =
        serde_json::from_str(include_str!("../src/ui/locales/en.json")).unwrap();
    keys.contains_key(text)
}

#[test]
fn every_tab_renders_in_both_languages_without_a_raw_key() {
    for (lang, atelier, graph) in [
        (Lang::Fr, "Atelier", "Graphe 3D"),
        (Lang::En, "Workshop", "3D Graph"),
    ] {
        let mut b = bench(lang);
        for tab in Tab::ALL {
            let texts = b.texts(tab);
            assert!(texts.iter().any(|t| t == atelier), "{lang:?} {tab:?}");
            assert!(texts.iter().any(|t| t == graph), "{lang:?} {tab:?}");
            let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
            assert!(raw.is_empty(), "{lang:?} {tab:?}: {raw:?}");
        }
    }
}

#[test]
fn the_language_switch_repaints_the_window() {
    let mut b = bench(Lang::Fr);
    assert!(b
        .texts(Tab::Analyses)
        .iter()
        .any(|t| t == "Lancer l'analyse"));
    b.app.desktop.set_lang(Lang::En);
    let texts = b.texts(Tab::Analyses);
    assert!(texts.iter().any(|t| t == "Run the analysis"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "Lancer l'analyse"));
}

#[test]
fn the_keyboard_reaches_tabs_palette_and_tree() {
    let mut b = bench(Lang::En);
    b.press(key(Key::Num2, Modifiers::COMMAND));
    assert_eq!(b.app.tab, Tab::Graph);
    b.press(key(Key::Num5, Modifiers::COMMAND));
    assert_eq!(b.app.tab, Tab::Diagnostics);

    // Ctrl+K, a query, Enter: the first match runs.
    b.press(key(Key::K, Modifiers::COMMAND));
    assert!(b.app.palette.open);
    b.press(vec![Event::Text("analyses".into())]);
    b.press(key(Key::Enter, Modifiers::NONE));
    assert!(!b.app.palette.open);
    assert_eq!(b.app.tab, Tab::Analyses);
    b.press(key(Key::K, Modifiers::COMMAND));
    b.press(key(Key::Escape, Modifiers::NONE));
    assert!(!b.app.palette.open);

    // Down walks the tree from the selection, Escape clears it.
    b.app.tab = Tab::Graph;
    let before = b.app.selected.clone();
    b.press(key(Key::ArrowDown, Modifiers::NONE));
    assert!(b.app.selected.is_some() && b.app.selected != before);
    b.press(key(Key::Escape, Modifiers::NONE));
    assert_eq!(b.app.selected, None);
}

#[test]
fn advance_puts_the_agent_in_read_only_and_the_atelier_on_the_audit() {
    let mut b = bench(Lang::Fr);
    assert_eq!(
        b.app.agents.agent.mode,
        AgentMode::Modify,
        "the demo starts in Modify"
    );
    let scalping = b.texts(Tab::Atelier);
    assert!(
        scalping.iter().any(|t| t == "CHIRURGIE · CIBLAGE"),
        "{scalping:?}"
    );
    b.app.set_mode(Mode::Advance);
    assert_eq!(b.app.agents.agent.mode, AgentMode::Ask);
    let advance = b.texts(Tab::Atelier);
    assert!(
        advance.iter().any(|t| t == "AUDIT IA · LECTURE SEULE"),
        "{advance:?}"
    );
    assert!(advance.iter().any(|t| t == "Lancer l'audit"));
}

#[test]
fn the_demo_waits_on_an_approval_in_the_atelier_and_in_agents() {
    let mut b = bench(Lang::En);
    for tab in [Tab::Atelier, Tab::Agents] {
        let texts = b.texts(tab);
        assert!(
            texts.iter().any(|t| t == "Allow once"),
            "{tab:?}: {texts:?}"
        );
        assert!(texts.iter().any(|t| t == "Deny"), "{tab:?}");
    }
}

#[test]
fn every_pair_of_colors_reaches_its_contrast() {
    let failures = theme::contrast_failures();
    assert!(failures.is_empty(), "{failures:#?}");
    let style = theme::style();
    let tokens = theme::token_colors();
    let v = &style.visuals;
    for color in [
        v.panel_fill,
        v.window_fill,
        v.extreme_bg_color,
        v.selection.bg_fill,
    ] {
        assert!(tokens.contains(&color), "{color:?} is not a token");
    }
}

/// The characters a TrueType face maps, from its `cmap` format 4 subtable (the BMP).
fn cmap(font: &[u8]) -> std::collections::BTreeSet<char> {
    let u16_at = |o: usize| u16::from_be_bytes([font[o], font[o + 1]]) as usize;
    let u32_at = |o: usize| u32::from_be_bytes(font[o..o + 4].try_into().unwrap()) as usize;
    let tables = u16_at(4);
    let cmap = (0..tables)
        .map(|k| 12 + 16 * k)
        .find(|&r| &font[r..r + 4] == b"cmap")
        .map(|r| u32_at(r + 8))
        .expect("a cmap table");
    let sub = (0..u16_at(cmap + 2))
        .map(|k| cmap + u32_at(cmap + 4 + 8 * k + 4))
        .find(|&o| u16_at(o) == 4)
        .expect("a format 4 subtable");
    let segs = u16_at(sub + 6) / 2;
    let (ends, starts) = (sub + 14, sub + 16 + 2 * segs);
    let mut out = std::collections::BTreeSet::new();
    for k in 0..segs {
        let (start, end) = (u16_at(starts + 2 * k), u16_at(ends + 2 * k));
        out.extend((start..=end.min(0xFFFE)).filter_map(|c| char::from_u32(c as u32)));
    }
    out
}

#[test]
fn the_embedded_faces_cover_french_and_the_symbols() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/fonts/");
    let face = |name: &str| cmap(&std::fs::read(format!("{dir}{name}")).unwrap());
    let mono = face("JetBrainsMono-Regular.ttf");
    for name in ["Barlow-Regular.ttf", "Barlow-Medium.ttf"] {
        let sans = face(name);
        let missing: String = fonts::FRENCH
            .chars()
            .filter(|c| !sans.contains(c))
            .collect();
        assert!(missing.is_empty(), "{name} lacks {missing:?}");
        // A symbol Barlow lacks falls back to the monospace face, which must have it.
        let nowhere: String = fonts::SYMBOLS
            .chars()
            .filter(|c| !sans.contains(c) && !mono.contains(c))
            .collect();
        assert!(
            nowhere.is_empty(),
            "{name} and the fallback lack {nowhere:?}"
        );
    }
    let missing: String = fonts::FRENCH
        .chars()
        .chain(fonts::SYMBOLS.chars())
        .filter(|c| !mono.contains(c))
        .collect();
    assert!(missing.is_empty(), "JetBrains Mono lacks {missing:?}");
}

/// Share of pixels within 24 (Chebyshev, RGB) of either blue, outside the data regions.
fn accent_share(image: &raster::Image, data: &[Rect]) -> f64 {
    let blues = [theme::TOKENS.color.accent, theme::TOKENS.color.action];
    let (mut hits, mut total) = (0u64, 0u64);
    for y in 0..image.height {
        for x in 0..image.width {
            let p = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
            if data.iter().any(|r| r.contains(p)) {
                continue;
            }
            total += 1;
            let i = ((y * image.width + x) * 4) as usize;
            let px = &image.rgba[i..i + 3];
            let near = blues.iter().any(|b| {
                let d = [b.r(), b.g(), b.b()]
                    .iter()
                    .zip(px)
                    .map(|(a, c)| a.abs_diff(*c))
                    .max()
                    .unwrap_or(0);
                d < 24
            });
            hits += u64::from(near);
        }
    }
    hits as f64 / total.max(1) as f64
}

#[test]
fn no_screen_spends_more_than_three_percent_of_its_pixels_on_the_accent() {
    let mut b = bench(Lang::Fr);
    let mut lines = Vec::new();
    for tab in Tab::ALL {
        b.app.tab = tab;
        let app = &mut b.app;
        let frame = raster::render(&b.ctx, SIZE, 1.0, |ui| app.draw(ui));
        let data: Vec<Rect> = components::take_dump(&b.ctx)
            .into_iter()
            .filter(|(name, _)| ["preview", "graph-view", "histogram"].contains(&name.as_str()))
            .map(|(_, r)| r)
            .collect();
        let share = accent_share(&frame.image, &data);
        lines.push(format!("{tab:?}: {:.2} %", share * 100.0));
        assert!(share < 0.03, "{tab:?}: {:.2} %", share * 100.0);
    }
    println!("{}", lines.join("\n"));
}
