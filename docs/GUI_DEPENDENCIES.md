# Dependencies of the desktop app (`crates/surgeon-gui`)

The engine crate (`weight-surgeon`, root `Cargo.toml`) stays pure Rust with no network stack,
no TLS, no GUI and no C build step: `tests/deps_purity.rs` reads the engine's own resolved
tree (`cargo tree -p weight-surgeon`, every platform) and fails on any of them. The desktop app
is a member of the root workspace, which has **one `Cargo.lock`** for the engine, the app and
the terminal inspector: everything below is in that lockfile but never in the engine's tree,
which is what the check reads. The engine is reused as a
library (`weight_surgeon::backend`, `graph`, `patch`, `hardware`, `app::run_with`): the GUI
duplicates none of its logic. The studio (projects, job manager, event stream) added no
dependency to the engine: `src/events.rs` only uses `serde`, `serde_json` and `schemars`, which
were already there.

## Added for the installer and the first-run wizard

| Crate | Why it is needed | Why not something smaller |
| --- | --- | --- |
| `ureq` 3 (`native-tls` on Windows, `rustls` elsewhere) | The only HTTP client: releases API, archive and model downloads, update check. HTTPS, redirects followed by hand (each hop is checked against the domain allow-list), `Range` for resumable downloads, system and environment proxies. | `std` has no TLS. `ureq` is blocking and small (no async runtime). On Windows it uses the system TLS stack (schannel): the system trust store, no C compiler, no second crypto library. |
| `zip` 9 (`deflate` only) | Reads the mistral.rs release archive (zip64, data descriptors, central directory). The extraction rules (no `..`, no absolute path or drive letter, no symlink, size and entry caps) are ours, in `src/extract.rs`, and checked on the whole archive before the first byte is written. | A hand-written zip reader would have to get the format's corner cases right; this crate is already in the lockfile through the engine's dependencies. |
| `sha2` | SHA-256 of every download (releases publish `digest: sha256:...`; the sample model hashes are pinned in `src/sample_model.rs`). | Same crate and version as the engine. |
| `png` | Writes the `--screenshot` PNG and decodes the window icon. | Already in the lockfile through the UI stack; no new crate. |
| `windows-sys` 0.61 (`Win32_System_Console`, Windows only) | One call, `AttachConsole`, so that a GUI-subsystem build still prints `--version` and `--self-test` into the shell that started it. The only `unsafe` of the binary, annotated in `src/main.rs`. | There is no safe std equivalent. |
| `rfd` 0.18 (no default feature, Windows only) | The system dialog that picks the source model of a new project (a weights file or a model folder) and the workspace folder (wizard step 03, settings), run off the UI thread (`platform::Picker`). | Writing `IFileOpenDialog` by hand means COM and `unsafe` the crate denies; on Windows rfd reuses the `windows-sys` 0.61 already locked and adds one crate. Elsewhere it would need GTK or a desktop portal, so the path field stays there. |
| `embed-resource` (build script, Windows only) | Compiles the manifest (per-monitor DPI, long paths, UTF-8 code page, no elevation), the icon and the version information into the `.exe`. | The alternative is a hand-managed `.rc` toolchain step. |
| `tempfile` (dev only) | Temporary folders in the tests. | Same crate and version as the engine's dev-dependencies. |

## Added for the studio (projects, jobs, ledger)

| Crate | Why it is needed | Cost |
| --- | --- | --- |
| `schemars` 0.8 | JSON schema of the project manifest (`docs/schemas/project.surgeon.schema.json`). | No new crate: already in this lockfile through the engine, same version. |
| `sysinfo` 0.40 (`disk`, `system`) | Free space of the volume before a working copy; whether the PID behind `project.lock` is alive. | No new crate: already in this lockfile through the engine; only the `disk` feature is added. |
| `tokenizers` (dev only, same features as the engine) | The mini Qwen2 of `examples/make_fake_qwen2.rs`, for a real engine job in the tests. | Already in the engine's dev-dependencies. |

`clap`, `serde`, `serde_json`, `eframe` / `egui` were already used by the viewer.

## Added for the agent hub

| Crate | Why it is needed | Cost |
| --- | --- | --- |
| `windows-sys` feature `Win32_Security_Credentials` (Windows only) | `CredWriteW`, `CredReadW`, `CredDeleteW`: API keys in the Windows Credential Manager (`src/agent/vault.rs`, the only other `unsafe` of the crate, annotated). | No new crate: a feature of a crate already used; no lockfile change. |

The providers reuse `ureq` and its TLS stack (schannel on Windows: no C build), kept in this
crate; the engine crate stays without network, TLS or keychain (`tests/deps_purity.rs`).

## What is *not* added, and why

- **No file-dialog crate**: the home screen takes a path (typed or dropped on the window).
- **No async runtime, no `reqwest`, no `tokio`**: downloads run on plain threads (`src/task.rs`).
- **No keychain crate**: the agent hub's API keys go to the Windows Credential Manager
  through three calls of `windows-sys` (below), never to `settings.json`, `agents.json` or
  the logs. Other systems keep the key for the session only, for now (docs/AGENTS.md).
- **No LLM SDK, no SSE crate**: the Anthropic and OpenAI streams are read by a small SSE
  reader over the same `ureq` (`src/agent/provider.rs`); there is no official Rust SDK.
- **No telemetry, no crash reporter, no auto-updater.** The update check is a button.

## Network surface

Allowed domains (`src/net.rs`, `ALLOWED_HOSTS`): `api.github.com`, `github.com`,
`*.githubusercontent.com`, `huggingface.co`, `*.huggingface.co`, `*.hf.co`. HTTPS only, no
credentials in URLs. Nothing is requested before the user clicks a button next to the URL,
size and hash; there is no background traffic. The studio itself (projects, jobs, ledger) makes
no network access.

The agent hub is the exception, and only after a consent per connection: it talks to the
endpoint the user configured (Anthropic, OpenAI, or any URL for a compatible server), HTTPS
only except to this machine (`localhost`, `127.0.0.1`, `::1`), no redirects, no credentials in
URLs. The "local only" switch refuses every non-loopback endpoint before a socket is opened.
The Claude Code and Codex CLIs, when used, make their own requests (docs/AGENTS.md).

## Reading the lockfile

The root `Cargo.lock` lists the packages of **every** member and platform, so it also names the
TLS stacks the Windows build never compiles (`rustls`, `ring`, `openssl-sys`,
`security-framework`...). On Windows the HTTP client is `ureq` + `native-tls` + `schannel`
(plus `http`, `httparse`, `ureq-proto`, `utf8-zero`); `embed-resource` pulls `vswhom` and
`winreg` to find the Windows SDK's resource compiler. Roughly thirty packages are new compared
with the viewer-only crate, none of them in the engine's tree. One package can be shared with
the engine and still be built differently for each: `zip` is the same lock entry, with
`deflate` for the app and no compression feature at all under `cargo tree -p weight-surgeon`.
