//! P12 guards: `unsafe` only in `src/kernel/**` (plus the engine's mmap), every block there
//! documented by `// SAFETY:`; no print macro, `unwrap`, `expect` nor `panic!` in the code
//! (non-test part) of the kernel and GGUF modules.

use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn src(sub: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    rust_files(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(sub),
        &mut files,
    );
    files
}

/// Code lines before the first `#[cfg(test)]`, without line comments.
fn code_lines(path: &Path) -> Vec<(usize, String)> {
    let source = std::fs::read_to_string(path).unwrap();
    let code = source.split("#[cfg(test)]").next().unwrap_or_default();
    code.lines()
        .enumerate()
        .map(|(n, l)| (n + 1, l.trim().to_owned()))
        .filter(|(_, l)| !l.starts_with("//"))
        .collect()
}

fn uses_unsafe(line: &str) -> bool {
    ["unsafe {", "unsafe fn", "unsafe impl", "unsafe extern"]
        .iter()
        .any(|k| line.contains(k))
}

fn rel(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[test]
fn unsafe_lives_only_in_the_kernel_and_the_engine_mmap() {
    for path in src("") {
        let name = rel(&path);
        if name.starts_with("src/kernel/") || name == "src/engine/mmap_file.rs" {
            continue;
        }
        for (n, line) in code_lines(&path) {
            assert!(!uses_unsafe(&line), "{name}:{n}: unsafe outside src/kernel");
        }
    }
}

#[test]
fn every_kernel_unsafe_block_has_a_safety_comment() {
    let mod_rs =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernel/mod.rs"))
            .unwrap();
    assert!(mod_rs.contains("#![deny(unsafe_op_in_unsafe_fn)]"));
    let mut blocks = 0;
    for path in src("kernel") {
        let source = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let code = line.trim();
            if code.starts_with("//") || !code.contains("unsafe {") {
                continue;
            }
            blocks += 1;
            // The comment sits on one of the few lines above (a statement may start with
            // `let x =` or a match arm before the block).
            let documented = lines[i.saturating_sub(6)..i]
                .iter()
                .any(|l| l.trim().starts_with("// SAFETY:"));
            assert!(
                documented,
                "{}:{}: unsafe block without // SAFETY:",
                rel(&path),
                i + 1
            );
        }
    }
    assert!(
        blocks >= 20,
        "only {blocks} unsafe blocks found: the scan is broken"
    );
}

#[test]
fn kernel_and_gguf_have_no_print_unwrap_or_panic() {
    let mut files = src("kernel");
    files.extend(src("patch/gguf"));
    for path in files {
        let name = rel(&path);
        // Test-only module (`#[cfg(test)] mod tests;` in kernel/mod.rs).
        if name == "src/kernel/tests.rs" {
            continue;
        }
        for (n, line) in code_lines(&path) {
            for banned in [
                "println!(",
                "print!(",
                "eprintln!(",
                "eprint!(",
                "dbg!(",
                ".unwrap()",
                ".expect(",
                "panic!(",
                "unimplemented!(",
                "todo!(",
            ] {
                assert!(!line.contains(banned), "{name}:{n}: {banned}");
            }
        }
    }
}
