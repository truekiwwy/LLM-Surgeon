//! The installer and release files are versioned, and keep the properties the product
//! promises: per-user (no administrator rights), clean uninstall, optional PATH, no silent
//! update, signing only when the secrets exist. The scripts are compiled for real in CI
//! (`.github/workflows/packaging-check.yml`, `release.yml`).

use std::path::PathBuf;

fn repo(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let text = std::fs::read_to_string(root.join(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.replace("\r\n", "\n")
}

#[test]
fn the_installer_is_per_user_and_uninstalls_cleanly() {
    let nsi = repo("packaging/windows/installer.nsi");
    assert!(nsi.contains("RequestExecutionLevel user"));
    assert!(nsi.contains("$LOCALAPPDATA\\Programs"));
    assert!(!nsi.contains("HKLM"), "nothing machine-wide");
    assert!(!nsi.contains("$PROGRAMFILES"), "nothing machine-wide");
    for needle in [
        "WriteUninstaller",
        "DeleteRegKey HKCU",
        "Delete \"$SMPROGRAMS",
        "surgeon.exe",
    ] {
        assert!(nsi.contains(needle), "{needle}");
    }
    // PATH is a section of its own, unchecked by default (/o).
    assert!(nsi.contains("Section /o \"Ajouter surgeon.exe au PATH"));
    // User data is only deleted after a question that defaults to No.
    assert!(nsi.contains("MB_YESNO|MB_DEFBUTTON2"));
}

#[test]
fn the_build_script_signs_only_with_secrets_and_ships_checksums() {
    let ps = repo("packaging/build-windows.ps1");
    assert!(ps.contains("SIGN_CERT_BASE64") && ps.contains("SIGN_CERT_PASSWORD"));
    assert!(
        ps.contains("are NOT signed"),
        "the unsigned case is explicit"
    );
    assert!(ps.contains("SHA256SUMS"));
    assert!(ps.contains("-DryRun"));
    assert!(ps.contains("portable.txt"));
}

#[test]
fn the_release_workflow_runs_on_version_tags_and_does_not_publish_by_itself() {
    let yml = repo(".github/workflows/release.yml");
    assert!(yml.contains("tags: [\"v*\"]"));
    assert!(yml.contains("windows-latest"));
    assert!(yml.contains("deps_purity"));
    assert!(
        yml.contains("--draft"),
        "a maintainer publishes the release"
    );
    let check = repo(".github/workflows/packaging-check.yml");
    assert!(check.contains("-DryRun"));
}

#[test]
fn the_binary_is_built_small_and_static() {
    // One workspace: the profiles live in the root manifest, `dist` is the GUI's.
    let manifest = repo("Cargo.toml");
    let dist = manifest.split("[profile.dist]").nth(1).unwrap_or_default();
    assert!(dist.contains("inherits = \"release\"") && dist.contains("lto = \"fat\""));
    assert!(manifest.contains("strip = true"));
    assert!(repo("packaging/build-windows.ps1").contains("cargo build --profile dist"));
    let cargo = repo("crates/surgeon-gui/.cargo/config.toml");
    assert!(cargo.contains("+crt-static"));
    let manifest_xml = repo("crates/surgeon-gui/assets/app.manifest");
    for needle in ["PerMonitorV2", "longPathAware", "UTF-8", "asInvoker"] {
        assert!(manifest_xml.contains(needle), "{needle}");
    }
}

#[test]
fn every_new_dependency_is_justified() {
    let doc = repo("docs/GUI_DEPENDENCIES.md");
    for krate in [
        "ureq",
        "zip",
        "sha2",
        "png",
        "windows-sys",
        "embed-resource",
        "tempfile",
    ] {
        assert!(
            doc.contains(krate),
            "{krate} is not in docs/GUI_DEPENDENCIES.md"
        );
    }
}
