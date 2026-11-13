# Dependencies of the desktop app (`crates/surgeon-gui`)

The engine crate (`weight-surgeon`, root `Cargo.toml`) stays pure Rust with no network stack,
no TLS, no GUI and no C build step: `tests/deps_purity.rs` reads the root `Cargo.lock` and
fails on any of them. The desktop app is a separate crate with **its own `Cargo.lock`**, so
everything below lives there and never enters the engine's lockfile. The engine is reused as a
library (`weight_surgeon::backend`, `graph`, `patch`, `hardware`, `app::run_with`): the GUI
duplicates none of its logic.

## Added for the installer and the first-run wizard

| Crate | Why it is needed | Why not something smaller |
| --- | --- | --- |
| `ureq` 3 (`native-tls` on Windows, `rustls` elsewhere) | The only HTTP client: releases API, archive and model downloads, update check. HTTPS, redirects followed by hand (each hop is checked against the domain allow-list), `Range` for resumable downloads, system and environment proxies. | `std` has no TLS. `ureq` is blocking and small (no async runtime). On Windows it uses the system TLS stack (schannel): the system trust store, no C compiler, no second crypto library. |
| `zip` 9 (`deflate` only) | Reads the llama.cpp release archives (zip64, data descriptors, central directory). The extraction rules (no `..`, no absolute path or drive letter, no symlink, size and entry caps) are ours, in `src/extract.rs`, and checked on the whole archive before the first byte is written. | A hand-written zip reader would have to get the format's corner cases right; this crate is already in the lockfile through the engine's dependencies. |
| `sha2` | SHA-256 of every download (releases publish `digest: sha256:...`; the sample model hashes are pinned in `src/sample_model.rs`). | Same crate and version as the engine. |
| `png` | Writes the `--screenshot` PNG and decodes the window icon. | Already in the lockfile through the UI stack; no new crate. |
| `windows-sys` 0.61 (`Win32_System_Console`, Windows only) | One call, `AttachConsole`, so that a GUI-subsystem build still prints `--version` and `--self-test` into the shell that started it. The only `unsafe` of the binary, annotated in `src/main.rs`. | There is no safe std equivalent. |
| `embed-resource` (build script, Windows only) | Compiles the manifest (per-monitor DPI, long paths, UTF-8 code page, no elevation), the icon and the version information into the `.exe`. | The alternative is a hand-managed `.rc` toolchain step. |
| `tempfile` (dev only) | Temporary folders in the tests. | Same crate and version as the engine's dev-dependencies. |

`clap`, `serde`, `serde_json`, `eframe` / `egui` were already used by the viewer.

## What is *not* added, and why

- **No file-dialog crate**: the home screen takes a path (typed or dropped on the window).
- **No async runtime, no `reqwest`, no `tokio`**: downloads run on plain threads (`src/task.rs`).
- **No keychain crate**: the app stores no secret. The sample model and the releases are
  public; nothing asks for a token. If one is ever needed it must go through the OS keychain,
  never `settings.json` or the logs (which mask secrets: `src/logging.rs`).
- **No telemetry, no crash reporter, no auto-updater.** The update check is a button.

## Network surface

Allowed domains (`src/net.rs`, `ALLOWED_HOSTS`): `api.github.com`, `github.com`,
`*.githubusercontent.com`, `huggingface.co`, `*.huggingface.co`, `*.hf.co`. HTTPS only, no
credentials in URLs. Nothing is requested before the user clicks a button next to the URL,
size and hash; there is no background traffic.

## Reading the lockfile

`crates/surgeon-gui/Cargo.lock` lists the packages of **every** platform, so it also names the
TLS stacks the Windows build never compiles (`rustls`, `ring`, `openssl-sys`,
`security-framework`...). On Windows the HTTP client is `ureq` + `native-tls` + `schannel`
(plus `http`, `httparse`, `ureq-proto`, `utf8-zero`); `embed-resource` pulls `vswhom` and
`winreg` to find the Windows SDK's resource compiler. Roughly thirty packages are new compared
with the viewer-only crate, none of them in the engine's lockfile.
