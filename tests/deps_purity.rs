//! The extraction stack stays pure Rust: no C/C++ build step, no GPU SDK, no Oniguruma.
//! Reads `Cargo.lock`, which lists every platform and the dev-dependencies too.

const FORBIDDEN: [&str; 13] = [
    "onig_sys",
    "onig",
    "cmake",
    "bindgen",
    "cudarc",
    "cuda-sys",
    "metal",
    "intel-mkl-src",
    "accelerate-src",
    "cc",
    "candle-kernels",
    "candle-metal-kernels",
    "candle-flash-attn",
];

/// `(name, dependencies)` of every `[[package]]` entry.
fn packages(lock: &str) -> Vec<(String, Vec<String>)> {
    lock.split("[[package]]")
        .skip(1)
        .filter_map(|entry| {
            let name = entry
                .lines()
                .find_map(|l| l.strip_prefix("name = \""))?
                .trim_end_matches('"')
                .to_owned();
            let deps = entry
                .split("dependencies = [")
                .nth(1)
                .map(|d| d.split(']').next().unwrap_or_default())
                .map(|d| {
                    d.lines()
                        .filter_map(|l| l.trim().strip_prefix('"'))
                        .map(|l| l.split([' ', '"']).next().unwrap_or_default().to_owned())
                        .collect()
                })
                .unwrap_or_default();
            Some((name, deps))
        })
        .collect()
}

#[test]
fn no_native_build_dependency_in_the_lockfile() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock")).unwrap();
    let packages = packages(&lock);
    assert!(
        packages.iter().any(|(n, _)| n == "candle-core"),
        "parser sanity check"
    );
    for (name, _) in &packages {
        assert!(
            !FORBIDDEN.contains(&name.as_str()),
            "native or GPU dependency in Cargo.lock: {name}"
        );
    }
    // `tokenizers` always depends on esaxx-rs; only its `cpp` feature (which pulls `cc`)
    // is native. It must stay a dependency-free pure-Rust build.
    if let Some((_, deps)) = packages.iter().find(|(n, _)| n == "esaxx-rs") {
        assert!(deps.is_empty(), "esaxx-rs built with native code: {deps:?}");
    }
}
