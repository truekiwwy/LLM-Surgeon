//! Shared by the P17 suites: `examples/fake_llama_server.rs`, built next to `surgeon` (a
//! run of one suite alone never uses a stale or missing copy).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

pub fn fake_server() -> &'static Path {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT.get_or_init(|| {
        let profile_dir = Path::new(env!("CARGO_BIN_EXE_surgeon")).parent().unwrap();
        let mut build = Command::new(env!("CARGO"));
        build
            .args(["build", "--example", "fake_llama_server"])
            .current_dir(env!("CARGO_MANIFEST_DIR"));
        if profile_dir.ends_with("release") {
            build.arg("--release");
        }
        let output = build.output().unwrap();
        assert!(
            output.status.success(),
            "cargo build --example fake_llama_server: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let exe = profile_dir
            .join("examples")
            .join(format!("fake_llama_server{}", std::env::consts::EXE_SUFFIX));
        assert!(exe.is_file(), "{} missing", exe.display());
        exe
    })
}

/// The name llama.cpp gives the server binary on this platform.
pub fn server_name() -> String {
    format!("llama-server{}", std::env::consts::EXE_SUFFIX)
}
