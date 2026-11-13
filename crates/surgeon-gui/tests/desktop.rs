//! The executable as a user runs it: arguments, off-screen capture, reset, logs. Every run
//! gets its own data folder and user config, so nothing depends on (or touches) the
//! machine's own profile. The graphics self-test needs an OpenGL driver and is not run here:
//! it is a manual step and a non-blocking CI step.

use std::path::Path;
use std::process::{Command, Output};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen2-mini.ndjson"
);

fn run(data: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_surgeon-gui"))
        .args(args)
        .env("SURGEON_GUI_DATA_DIR", data)
        .env("WEIGHT_SURGEON_CONFIG", data.join("config.toml"))
        .env("HF_TOKEN", "hf_this_must_never_be_logged")
        .output()
        .expect("spawn surgeon-gui")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn is_png(path: &Path) -> bool {
    std::fs::read(path).is_ok_and(|b| b.starts_with(b"\x89PNG"))
}

#[test]
fn version_is_printed() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(dir.path(), &["--version"]);
    assert!(out.status.success());
    assert!(stdout(&out).contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn no_argument_renders_the_atelier_off_screen() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("atelier.png");
    let out = run(dir.path(), &["--screenshot", png.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(is_png(&png));
}

#[test]
fn legacy_graph_arguments_still_open() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("graph.png");
    let out = run(
        dir.path(),
        &["--screenshot", png.to_str().unwrap(), FIXTURE],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(is_png(&png));
    // A path that does not exist is an error, not a crash.
    let out = run(
        dir.path(),
        &["--screenshot", png.to_str().unwrap(), "does-not-exist.json"],
    );
    assert!(!out.status.success());
}

#[test]
fn every_screen_can_be_captured_and_unknown_ones_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    for screen in [
        "atelier",
        "graph",
        "analyses",
        "agents",
        "diagnostics",
        "wizard",
    ] {
        let png = dir.path().join(format!("{screen}.png"));
        let out = run(
            dir.path(),
            &["--screenshot", png.to_str().unwrap(), "--screen", screen],
        );
        assert!(
            out.status.success(),
            "{screen}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(is_png(&png), "{screen}");
    }
    let png = dir.path().join("x.png");
    assert!(!run(
        dir.path(),
        &["--screenshot", png.to_str().unwrap(), "--screen", "home"]
    )
    .status
    .success());
}

#[test]
fn the_demo_workspace_renders_every_tab_at_both_sizes_and_scales() {
    let dir = tempfile::tempdir().unwrap();
    for (screen, size, scale) in [
        ("atelier", "1440x900", "1.0"),
        ("agents", "1920x1080", "1.0"),
        ("graph", "1440x900", "1.5"),
    ] {
        let png = dir.path().join(format!("{screen}.png"));
        let out = run(
            dir.path(),
            &[
                "--screenshot",
                png.to_str().unwrap(),
                "--screen",
                screen,
                "--demo-data",
                "--no-motion",
                "--size",
                size,
                "--scale",
                scale,
            ],
        );
        assert!(
            out.status.success(),
            "{screen}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let bytes = std::fs::read(&png).unwrap();
        assert!(is_png(&png), "{screen}");
        // IHDR: width and height, big endian, right after the signature and chunk header.
        let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(format!("{w}x{h}"), size, "{screen}");
    }
}

#[test]
fn a_run_replays_over_the_graph_and_a_bad_event_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let events = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/events/v1.ndjson"
    );
    let png = dir.path().join("replay.png");
    let png = png.to_str().unwrap();
    let args = |events| {
        [
            "--screenshot",
            png,
            "--screen",
            "graph",
            "--replay",
            events,
            FIXTURE,
        ]
    };
    let out = run(dir.path(), &args(events));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(is_png(Path::new(png)));
    let bad = dir.path().join("bad.ndjson");
    std::fs::write(&bad, "garbage\n").unwrap();
    let out = run(dir.path(), &args(bad.to_str().unwrap()));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("ligne 1"));
}

#[test]
fn reset_settings_forgets_the_wizard_but_keeps_components() {
    let dir = tempfile::tempdir().unwrap();
    let settings = dir.path().join("settings.json");
    let component = dir
        .path()
        .join("components")
        .join("llama.cpp")
        .join("kept.txt");
    std::fs::create_dir_all(component.parent().unwrap()).unwrap();
    std::fs::write(&component, "x").unwrap();
    std::fs::write(&settings, r#"{"wizard_done":true}"#).unwrap();
    let png = dir.path().join("p.png");
    let out = run(
        dir.path(),
        &["--reset-settings", "--screenshot", png.to_str().unwrap()],
    );
    assert!(out.status.success());
    assert!(!settings.exists(), "settings.json is removed");
    assert!(component.exists(), "installed components are kept");
    // Resetting twice is fine.
    assert!(run(
        dir.path(),
        &["--reset-settings", "--screenshot", png.to_str().unwrap()]
    )
    .status
    .success());
}

#[test]
fn no_secret_reaches_the_logs_or_the_data_folder() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("d.png");
    run(
        dir.path(),
        &[
            "--screenshot",
            png.to_str().unwrap(),
            "--screen",
            "diagnostics",
        ],
    );
    let mut stack = vec![dir.path().to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(&path).unwrap().flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_none_or(|e| e != "png") {
                let text = String::from_utf8_lossy(&std::fs::read(&p).unwrap()).into_owned();
                assert!(
                    !text.contains("hf_this_must_never_be_logged"),
                    "{}",
                    p.display()
                );
            }
        }
    }
}
