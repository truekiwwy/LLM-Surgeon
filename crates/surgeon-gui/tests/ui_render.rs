//! The real window, rendered off screen on the `--demo-data` workspace: both views, the
//! new-session form and every section of the settings in both languages, the keyboard
//! (views, settings, palette, tree), both kinds of session, the theme's contrasts and type
//! coverage, and the accent budget of each screen (docs/design/DIRECTION.md, section 2).

use eframe::egui::{self, Event, Key, Modifiers, Pos2, RawInput, Rect, Vec2};
use surgeon_gui::actions::Env;
use surgeon_gui::agent::policy::AgentMode;
use surgeon_gui::app::{Mode, RightPane, Section, SurgeonApp, Tab};
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

/// The demo session waits on an approval, so New session is refused: drop it by hand.
fn empty_session(b: &mut Bench) {
    b.app.agents.session = None;
    b.app.agents.cli_conv = None;
    assert!(b.app.agents.conversation().is_none());
}

/// The texts of one frame of the settings dialog on `section`, over the agent.
fn settings_texts(b: &mut Bench, section: Section) -> Vec<String> {
    b.app.settings = Some(section);
    b.texts(Tab::Agent)
}

#[test]
fn every_view_and_section_renders_in_both_languages_without_a_raw_key() {
    for (lang, evidence, inspector) in [
        (Lang::Fr, "Preuves", "Inspecteur"),
        (Lang::En, "Evidence", "Inspector"),
    ] {
        let mut b = bench(lang);
        let mut screens = vec![
            ("agent", b.texts(Tab::Agent)),
            ("graph", b.texts(Tab::Graph)),
        ];
        assert!(screens[0].1.iter().any(|t| t == evidence), "{lang:?}");
        assert!(screens[1].1.iter().any(|t| t == inspector), "{lang:?}");
        empty_session(&mut b);
        screens.push(("form", b.texts(Tab::Agent)));
        for section in Section::ALL {
            screens.push(("settings", settings_texts(&mut b, section)));
        }
        b.app.settings = None;
        for (name, texts) in screens {
            let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
            assert!(raw.is_empty(), "{lang:?} {name}: {raw:?}");
        }
    }
}

#[test]
fn the_radar_compares_two_direction_files_of_work() {
    for (lang, title, strong) in [
        (Lang::Fr, "Radar d'interférence", "Forte collision"),
        (Lang::En, "Interference radar", "Strong collision"),
    ] {
        let mut b = bench(lang);
        let project = b.app.studio.manager.as_ref().unwrap().project.dir.clone();
        let work = project.join("work");
        std::fs::create_dir_all(&work).unwrap();
        let meta = [("layer".to_owned(), "14".to_owned())].into();
        for (name, v) in [("a.safetensors", [1.0, 0.0]), ("b.safetensors", [0.8, 0.6])] {
            let bytes = weight_surgeon::extract::encode_tensors(&[("direction", &v)], &meta);
            std::fs::write(work.join(name), bytes.unwrap()).unwrap();
        }
        b.app.radar.open_on(&work);
        let texts = b.texts(Tab::Agent);
        // The layer comes from the metadata, the row reads in full inside the panel.
        for text in [title, "L14 · direction", "+0.8000", strong] {
            assert!(texts.iter().any(|t| t == text), "{text}: {texts:?}");
        }
        let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
        assert!(raw.is_empty(), "{lang:?}: {raw:?}");
    }
}

#[test]
fn the_vector_dialog_and_the_causal_scores_of_a_layer_render_in_both_languages() {
    use surgeon_gui::agent::session::Dirs;
    use surgeon_gui::studio::vectors::VectorMode;

    for (lang, title, run, causal, inside) in [
        (
            Lang::Fr,
            "Extraire un vecteur",
            "Lancer",
            "Score causal",
            "dedans",
        ),
        (
            Lang::En,
            "Extract a vector",
            "Run",
            "Causal score",
            "inside",
        ),
    ] {
        let mut b = bench(lang);
        b.app.vector.open = true;
        let sweep = b.texts(Tab::Agent);
        for text in [title, run, "pca (svd)", "work/direction.safetensors"] {
            assert!(sweep.iter().any(|t| t == text), "{text}: {sweep:?}");
        }
        b.app.vector.mode = VectorMode::Trace;
        let trace = b.texts(Tab::Agent);
        assert!(trace.iter().any(|t| t == "mlp"), "{trace:?}");
        let raw: Vec<_> = sweep.iter().chain(&trace).filter(|t| raw_key(t)).collect();
        assert!(raw.is_empty(), "{lang:?}: {raw:?}");
        b.app.vector.open = false;

        let project = b.app.agents.project.clone().unwrap();
        let data = serde_json::json!({"component": "mlp",
            "window": {"start": 1, "end": 2, "layers": "1-2"},
            "layers": [{"layer": 0, "score": 0.1}, {"layer": 1, "score": 0.75},
                       {"layer": 2, "score": 0.5}]});
        let line = serde_json::json!({"run_id": "mcp-1", "tool": "surgeon_causal_trace",
            "ok": true, "exit_code": 0, "data": data, "started_unix": 1, "finished_unix": 2});
        std::fs::write(Dirs::of(&project).mcp_runs(), format!("{line}\n")).unwrap();
        let step = b.app.timeline.current().unwrap();
        let node = step
            .report
            .nodes
            .iter()
            .find(|n| n.layer == Some(1))
            .unwrap();
        b.app.selected = Some(node.id.clone());
        assert_eq!(b.app.layer_scores().window, Some((1, 2)));
        let inspector = b.texts(Tab::Graph);
        for text in [causal, "+0.750 · mlp", inside] {
            assert!(inspector.iter().any(|t| t == text), "{text}: {inspector:?}");
        }
    }
}

#[test]
fn the_results_panel_shows_the_ollama_command_of_the_working_copy() {
    use surgeon_gui::agent::{paths::ProjectPaths, session::Dirs};
    use surgeon_gui::studio::ollama;

    for (lang, button) in [
        (Lang::Fr, "Exporter vers Ollama"),
        (Lang::En, "Export to Ollama"),
    ] {
        let mut b = bench(lang);
        let project = b.app.agents.project.clone().unwrap();
        // A measuring run, so that the panel lists something and offers its exports.
        let line = serde_json::json!({"run_id": "mcp-1", "tool": "surgeon_extract", "ok": true,
            "exit_code": 0, "data": null, "started_unix": 1, "finished_unix": 2});
        std::fs::write(Dirs::of(&project).mcp_runs(), format!("{line}\n")).unwrap();
        let paths = ProjectPaths::of(&project);
        let (copy, origin) = (paths.copy.as_deref(), paths.origin.as_deref());
        // The demo's copy holds the weights only; a Hugging Face folder has its config.
        let config = copy.unwrap().join("config.json");
        std::fs::write(config, r#"{"max_position_embeddings": 4096}"#).unwrap();
        let export = ollama::export(&project, "demo-qwen2", copy, origin).unwrap();
        let export = export.expect("the working copy is a SafeTensors folder");
        let command = export.command.clone();
        b.app.ollama = Some(Ok(export));
        b.app.right = RightPane::Results;
        let texts = b.texts(Tab::Agent);
        for text in [button, "SafeTensors · c1", command.as_str()] {
            assert!(texts.iter().any(|t| t == text), "{text}: {texts:?}");
        }
        let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
        assert!(raw.is_empty(), "{lang:?}: {raw:?}");
    }
}

#[test]
fn the_checkpoints_panel_shows_the_tree_the_front_and_the_restore_button() {
    use std::collections::BTreeMap;
    use surgeon_gui::agent::paths::ProjectPaths;

    for (lang, title, restore) in [
        (Lang::Fr, "Checkpoints & Pareto", "Restaurer ce checkpoint"),
        (Lang::En, "Checkpoints & Pareto", "Restore this checkpoint"),
    ] {
        let mut b = bench(lang);
        let project = b.app.agents.project.clone().unwrap();
        let copy = ProjectPaths::of(&project).copy.unwrap();
        let store = weight_surgeon::checkpoint::Store::in_work(&project.join("work"));
        let metrics = |target: f64, gain: f64, drift: f64| {
            BTreeMap::from([
                ("target".into(), target),
                ("gain".into(), gain),
                ("drift".into(), drift),
            ])
        };
        let base = store.create(&copy, "base", metrics(0.5, 0.0, 0.0)).unwrap();
        store
            .create(&copy, "v L4 a1", metrics(0.8, 0.3, 0.2))
            .unwrap();
        store.restore(&base.id, &copy).unwrap();
        b.app.right = RightPane::Checkpoints;
        b.app.checkpoint_selected = Some("0002".into());
        let texts = b.texts(Tab::Agent);
        for text in [
            title,
            restore,
            "0001",
            "0002",
            "v L4 a1 · +0.300 / 0.200",
            "80.0 %",
            "50.0 %",
        ] {
            assert!(
                texts.iter().any(|t| t == text),
                "{lang:?} {text}: {texts:?}"
            );
        }
        let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
        assert!(raw.is_empty(), "{lang:?}: {raw:?}");
    }
}

#[test]
fn live_mode_puts_the_session_beside_the_graph_under_its_telemetry() {
    for (lang, expected) in [
        (
            Lang::Fr,
            [
                "Télémétrie Live",
                "Où",
                "Quand",
                "Comment",
                "Vers où",
                "3. Compose",
            ],
        ),
        (
            Lang::En,
            [
                "Live telemetry",
                "Where",
                "When",
                "How",
                "Toward",
                "3. Compose",
            ],
        ),
    ] {
        let mut b = bench(lang);
        b.app.live = true;
        b.app.ghost = true;
        let texts = b.texts(Tab::Agent);
        // The demo agent waits on surgeon_apply of patch.json: the compose stage, its method
        // and the approval card the user must still reach.
        for text in expected.iter().chain(&["patch.json", "surgeon_apply"]) {
            assert!(texts.iter().any(|t| t == text), "{text}: {texts:?}");
        }
        let approve = match lang {
            Lang::Fr => "Autoriser une fois",
            Lang::En => "Allow once",
        };
        assert!(texts.iter().any(|t| t == approve), "{texts:?}");
        let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
        assert!(raw.is_empty(), "{lang:?}: {raw:?}");
        // A tab of the activity bar leaves Live mode for that tab.
        b.press(key(Key::Num2, Modifiers::COMMAND));
        assert!(!b.app.live);
        assert_eq!(b.app.tab, Tab::Graph);
    }
}

#[test]
fn the_advanced_panel_forces_a_merge_method_with_its_own_settings() {
    use surgeon_gui::agent::mission::Merge;

    for (lang, merge) in [(Lang::Fr, "Fusion"), (Lang::En, "Merge")] {
        let mut b = bench(lang);
        empty_session(&mut b);
        b.app.agents.forced_on = true;
        b.app.agents.forced.merge = Merge::Dare;
        // The panel sits below the fold of the form: a taller window shows all of it.
        let tall = |b: &mut Bench| {
            let app = &mut b.app;
            raster::render(&b.ctx, [1440, 1800], 1.0, |ui| app.draw(ui)).texts
        };
        let texts = tall(&mut b);
        assert!(!texts.iter().any(|t| t.contains("widget ID")), "{texts:?}");
        let all = [
            merge,
            "auto",
            "Orthogonal f64",
            "TIES-Merge",
            "DARE",
            "drop_rate",
            "seed",
        ];
        for text in all {
            assert!(texts.iter().any(|t| t == text), "{text}: {texts:?}");
        }
        assert!(!texts.iter().any(|t| t == "density"), "TIES only");
        b.app.agents.forced.merge = Merge::Ties;
        let texts = tall(&mut b);
        assert!(texts.iter().any(|t| t == "density"), "{texts:?}");
        let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
        assert!(raw.is_empty(), "{lang:?}: {raw:?}");
    }
}

#[test]
fn the_language_switch_repaints_the_window() {
    let mut b = bench(Lang::Fr);
    assert!(b.texts(Tab::Agent).iter().any(|t| t == "Preuves"));
    b.app.desktop.set_lang(Lang::En);
    let texts = b.texts(Tab::Agent);
    assert!(texts.iter().any(|t| t == "Evidence"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "Preuves"));
}

#[test]
fn the_keyboard_reaches_views_settings_palette_and_tree() {
    let mut b = bench(Lang::En);
    b.press(key(Key::Num2, Modifiers::COMMAND));
    assert_eq!(b.app.tab, Tab::Graph);
    b.press(key(Key::Num1, Modifiers::COMMAND));
    assert_eq!(b.app.tab, Tab::Agent);
    b.press(key(Key::B, Modifiers::COMMAND));
    assert!(b.app.columns.collapsed, "Ctrl+B folds the left column");
    assert!(b.app.desktop.settings.left_collapsed, "and keeps it");
    b.press(key(Key::B, Modifiers::COMMAND));
    assert!(!b.app.columns.collapsed);

    // Ctrl+, opens the settings; Escape closes them.
    b.press(key(Key::Comma, Modifiers::COMMAND));
    assert_eq!(b.app.settings, Some(Section::General));
    b.press(vec![]);
    b.press(key(Key::Escape, Modifiers::NONE));
    assert_eq!(b.app.settings, None);

    // Ctrl+K, a query, Enter: the first match runs.
    b.press(key(Key::K, Modifiers::COMMAND));
    assert!(b.app.palette.open);
    b.press(vec![Event::Text("graph".into())]);
    b.press(key(Key::Enter, Modifiers::NONE));
    assert!(!b.app.palette.open);
    assert_eq!(b.app.tab, Tab::Graph);
    b.press(key(Key::K, Modifiers::COMMAND));
    b.press(key(Key::Escape, Modifiers::NONE));
    assert!(!b.app.palette.open);

    // Down walks the tree from the selection, Escape clears it.
    let before = b.app.selected.clone();
    b.press(key(Key::ArrowDown, Modifiers::NONE));
    assert!(b.app.selected.is_some() && b.app.selected != before);
    b.press(key(Key::Escape, Modifiers::NONE));
    assert_eq!(b.app.selected, None);
}

#[test]
fn a_new_session_offers_the_full_scan_and_the_precise_problem() {
    let mut b = bench(Lang::En);
    empty_session(&mut b);
    b.app.set_mode(Mode::Advance);
    let advance = b.texts(Tab::Agent);
    for text in ["Full scan (Advance)", "Precise problem (Scalping)", "Start"] {
        assert!(advance.iter().any(|t| t == text), "{text}: {advance:?}");
    }
    assert!(!advance.iter().any(|t| t == "Excessive refusals"));
    b.app.set_mode(Mode::Scalping);
    assert_eq!(b.app.desktop.settings.mode, Mode::Scalping, "kept");
    assert_eq!(
        b.app.agents.agent.mode,
        AgentMode::Modify,
        "the contract is not the kind of session"
    );
    let scalping = b.texts(Tab::Agent);
    for text in [
        "Excessive refusals",
        "Repetitions and loops",
        "Describe the problem",
    ] {
        assert!(scalping.iter().any(|t| t == text), "{text}: {scalping:?}");
    }
}

/// One frame of the window, and the rectangle the layout dump calls `name`.
fn frame_and(b: &mut Bench, name: &str) -> (raster::Image, Rect) {
    let app = &mut b.app;
    let frame = raster::render(&b.ctx, SIZE, 1.0, |ui| app.draw(ui));
    let rect = components::take_dump(&b.ctx)
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, r)| r)
        .unwrap_or_else(|| panic!("no {name} in the layout dump"));
    (frame.image, rect)
}

#[test]
fn a_long_description_scrolls_inside_its_field_and_covers_nothing_below() {
    let mut b = bench(Lang::Fr);
    empty_session(&mut b);
    b.app.tab = Tab::Agent;
    b.app.set_mode(Mode::Advance);
    // The first frame of the form settles its layout; the second is the reference.
    frame_and(&mut b, "multiline");
    let (empty, field) = frame_and(&mut b, "multiline");
    b.app.agents.input = (1..=60)
        .map(|k| {
            format!(
                "ligne {k} du problème décrit
"
            )
        })
        .collect();
    let (long, grown) = frame_and(&mut b, "multiline");
    assert_eq!(field, grown, "the field keeps its size");
    // The band under the field (ideas, what the agent may do, Start) is pixel for pixel the same.
    let band = (field.bottom() as u32 + 2)..(field.bottom() as u32 + 160);
    let width = empty.width as usize;
    let changed: Vec<(u32, u32)> = band
        .flat_map(|y| (0..empty.width).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let i = (y as usize * width + x as usize) * 4;
            empty.rgba[i..i + 4] != long.rgba[i..i + 4]
        })
        .collect();
    let span = |f: fn(&(u32, u32)) -> u32| {
        let v: Vec<u32> = changed.iter().map(f).collect();
        (v.iter().min().copied(), v.iter().max().copied())
    };
    assert!(
        changed.is_empty(),
        "the text drew over what is below its field: x {:?}, y {:?}",
        span(|p| p.0),
        span(|p| p.1)
    );
}

#[test]
fn the_demo_waits_on_an_approval_in_the_thread_only() {
    let mut b = bench(Lang::En);
    let texts = b.texts(Tab::Agent);
    assert!(texts.iter().any(|t| t == "Allow once"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "Deny"));
    for text in ["Sessions", "Evidence"] {
        assert!(texts.iter().any(|t| t == text), "{text}: {texts:?}");
    }
    let graph = b.texts(Tab::Graph);
    assert!(!graph.iter().any(|t| t == "Allow once"), "{graph:?}");
    assert!(b.app.agents.pending() > 0, "the activity bar counts it");
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
    for name in ["DMSans-Regular.ttf", "DMSans-Medium.ttf", "DMSans-Bold.ttf"] {
        let sans = face(name);
        let missing: String = fonts::FRENCH
            .chars()
            .filter(|c| !sans.contains(c))
            .collect();
        assert!(missing.is_empty(), "{name} lacks {missing:?}");
        // A symbol DM Sans lacks falls back to the monospace face, which must have it.
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

/// Share of pixels within 24 (Chebyshev, RGB) of the accent, outside the data regions.
fn accent_share(image: &raster::Image, data: &[Rect]) -> f64 {
    let blues = [theme::TOKENS.color.accent];
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
            .filter(|(name, _)| name == "graph-view")
            .map(|(_, r)| r)
            .collect();
        let share = accent_share(&frame.image, &data);
        lines.push(format!("{tab:?}: {:.2} %", share * 100.0));
        assert!(share < 0.03, "{tab:?}: {:.2} %", share * 100.0);
    }
    println!("{}", lines.join("\n"));
}

#[test]
fn the_results_panel_offers_the_lora_export_and_shows_its_fidelity() {
    use surgeon_gui::agent::session::Dirs;

    for (lang, button) in [
        (Lang::Fr, "Extraire en LoRA (.safetensors)"),
        (Lang::En, "Extract as LoRA (.safetensors)"),
    ] {
        let mut b = bench(lang);
        let project = b.app.agents.project.clone().unwrap();
        let lora = serde_json::json!({"rank": 16, "tensors": 196, "mean_rel_error": 0.013,
            "out": "out/lora-r16/adapter_model.safetensors",
            "per_tensor": [{"name": "model.layers.3.mlp.down_proj.weight", "rel_error": 0.041}]});
        let line = serde_json::json!({"run_id": "mcp-3", "tool": "surgeon_lora", "ok": true,
            "exit_code": 0, "data": {"lora": lora}, "started_unix": 1, "finished_unix": 2});
        std::fs::write(Dirs::of(&project).mcp_runs(), format!("{line}\n")).unwrap();
        b.app.right = RightPane::Results;
        let texts = b.texts(Tab::Agent);
        for text in [button, "r = 16", "lora · r16 · 98.7 %"] {
            assert!(texts.iter().any(|t| t == text), "{text}: {texts:?}");
        }
        let raw: Vec<_> = texts.iter().filter(|t| raw_key(t)).collect();
        assert!(raw.is_empty(), "{lang:?}: {raw:?}");
    }
}
