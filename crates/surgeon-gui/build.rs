//! Windows resources: the manifest (DPI-aware, long paths, UTF-8 code page, no elevation),
//! the icon and the version information of the executable. The .rc is generated here so
//! that the version comes from Cargo.toml.

use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let assets = manifest_dir.join("assets");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let mut parts = version
        .split(['.', '-', '+'])
        .map(|p| p.parse::<u32>().unwrap_or(0));
    let (major, minor, patch) = (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    );
    let path = |name: &str| assets.join(name).display().to_string().replace('\\', "/");
    let rc = format!(
        r#"#pragma code_page(65001)
1 ICON "{icon}"
1 24 "{manifest}"
1 VERSIONINFO
FILEVERSION {major},{minor},{patch},0
PRODUCTVERSION {major},{minor},{patch},0
FILEOS 0x40004
FILETYPE 0x1
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "weight-surgeon"
      VALUE "FileDescription", "LLM Surgery Studio"
      VALUE "FileVersion", "{version}"
      VALUE "InternalName", "surgeon-gui"
      VALUE "LegalCopyright", "weight-surgeon contributors"
      VALUE "OriginalFilename", "surgeon-gui.exe"
      VALUE "ProductName", "LLM Surgery Studio"
      VALUE "ProductVersion", "{version}"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#,
        icon = path("icon.ico"),
        manifest = path("app.manifest"),
    );
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("app.rc");
    fs::write(&out, rc).expect("write app.rc");
    embed_resource::compile(&out, embed_resource::NONE)
        .manifest_required()
        .expect("compile the Windows resources");
}
