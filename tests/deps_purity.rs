//! The engine stays pure Rust: no C/C++ build step, no GPU SDK, no Oniguruma. Read from the
//! engine's own resolved tree (`cargo tree -p weight-surgeon`, every platform, build and dev
//! dependencies included), not from `Cargo.lock`: the workspace's single lockfile also lists
//! the desktop app's packages (windowing, TLS, the Windows resource compiler) and merges the
//! features its members ask for, none of which the engine builds.

const FORBIDDEN: [&str; 23] = [
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
    // Compression backends with C code (zip, pulled by candle-core for .pt files).
    "zstd-sys",
    "bzip2-sys",
    "libz-sys",
    "libz-ng-sys",
    "lzma-sys",
    // BLAS / LAPACK backends (Fortran or C): linalg is pure Rust.
    "openblas-src",
    "blas-src",
    "lapack-src",
    "netlib-src",
    "cblas-sys",
];

/// One node of the tree: its depth and its package name. A node seen before is not expanded
/// again by `cargo tree`.
struct Node {
    depth: usize,
    name: String,
}

/// The engine's dependency tree, as `cargo tree` resolves it for this package alone.
fn engine_tree() -> Vec<Node> {
    let out = std::process::Command::new(env!("CARGO"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "tree",
            "--locked",
            "-p",
            "weight-surgeon",
            "--target",
            "all",
            "-e",
            "normal,build,dev",
            "--prefix",
            "depth",
            "-f",
            "{p}",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cargo tree: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            let digits = line.chars().take_while(char::is_ascii_digit).count();
            Node {
                depth: line[..digits].parse().unwrap(),
                name: line[digits..].split(' ').next().unwrap().to_owned(),
            }
        })
        .collect()
}

/// The names of the direct dependencies of the first expansion of `name`.
fn children(tree: &[Node], name: &str) -> Option<Vec<String>> {
    let at = tree.iter().position(|n| n.name == name)?;
    let depth = tree[at].depth;
    Some(
        tree[at + 1..]
            .iter()
            .take_while(|n| n.depth > depth)
            .filter(|n| n.depth == depth + 1)
            .map(|n| n.name.clone())
            .collect(),
    )
}

#[test]
fn no_native_build_dependency_in_the_engine() {
    let tree = engine_tree();
    assert!(
        tree.iter().any(|n| n.name == "candle-core"),
        "parser sanity check"
    );
    for node in &tree {
        assert!(
            !FORBIDDEN.contains(&node.name.as_str()),
            "native or GPU dependency of the engine: {}",
            node.name
        );
    }
    // `tokenizers` always depends on esaxx-rs; only its `cpp` feature (which pulls `cc`)
    // is native. It must stay a dependency-free pure-Rust build.
    if let Some(deps) = children(&tree, "esaxx-rs") {
        assert!(deps.is_empty(), "esaxx-rs built with native code: {deps:?}");
    }
}

/// `bench logits` reads `.pt` files through candle-core's `zip`, built without default
/// features: no compression backend at all (only stored entries, which `torch.save`
/// writes), so neither C code nor a Rust inflater is linked for it. The desktop app's own
/// `zip` (deflate) is the same package in the lockfile, never in this tree.
#[test]
fn zip_has_no_compression_backend() {
    let tree = engine_tree();
    let Some(deps) = children(&tree, "zip") else {
        return;
    };
    for backend in [
        "flate2",
        "miniz_oxide",
        "zstd",
        "bzip2",
        "lzma-rs",
        "liblzma",
        "xz2",
        "deflate64",
        "zopfli",
    ] {
        assert!(
            !deps.iter().any(|d| d == backend),
            "zip built with {backend}: {deps:?}"
        );
    }
}
