//! P13 source guards: no print, no unwrap/expect/panic, no unsafe in the new modules
//! (test modules excluded), and no C-backed compression crate.

use std::path::{Path, PathBuf};

const ROOTS: [&str; 6] = [
    "src/surgery",
    "src/lora",
    "src/extract/generate.rs",
    "src/tui/tabs",
    "src/tui/preview_args.rs",
    "src/mcp/tools_p13.rs",
];

const BANNED: [&str; 7] = [
    "println!(",
    "print!(",
    "eprintln!(",
    "eprint!(",
    ".unwrap()",
    ".expect(",
    "panic!(",
];

fn rust_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        out.push(path.to_path_buf());
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() || p.extension().is_some_and(|e| e == "rs") {
            rust_files(&p, out);
        }
    }
}

#[test]
fn p13_modules_have_no_print_unwrap_panic_or_unsafe() {
    let mut files = Vec::new();
    for root in ROOTS {
        rust_files(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join(root),
            &mut files,
        );
    }
    assert!(!files.is_empty());
    for path in files {
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(!source.contains("unsafe"), "{}: unsafe", path.display());
        let code = source.split("#[cfg(test)]").next().unwrap_or_default();
        for (n, line) in code.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for banned in BANNED {
                assert!(
                    !line.contains(banned),
                    "{}:{}: {banned}",
                    path.display(),
                    n + 1
                );
            }
        }
    }
}

#[test]
fn compression_is_pure_rust() {
    let manifest =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .unwrap_or_default();
    assert!(manifest.contains("miniz_oxide"));
    assert!(!manifest.contains("zstd"));
}
