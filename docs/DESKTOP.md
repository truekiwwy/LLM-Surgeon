# Desktop app (Windows): install, first run, troubleshooting, uninstall

`surgeon-gui.exe` is LLM Surgery Studio: an AI agent in an IDE-like window (a full scan of
the model, or one precise problem), a first-run wizard that checks the machine and installs
mistral.rs, the 3D weight graph and a settings dialog (AI providers, engine, dependencies).
It runs the same engine as the `surgeon` command line (the same library, never a copy of it).

## Install

Two packages are published with each release, plus `SHA256SUMS`:

| File | What it is |
| --- | --- |
| `weight-surgeon-<version>-windows-x64-setup.exe` | Per-user installer. **No administrator rights.** Installs to `%LOCALAPPDATA%\Programs\LLM Surgery Studio`, adds a Start menu shortcut, optionally a desktop shortcut and, **unchecked by default**, adds `surgeon.exe` to your user `PATH`. Uninstall from Windows *Installed apps*. |
| `weight-surgeon-<version>-windows-x64.zip` | Portable. Unzip anywhere. `portable.txt` next to the executable keeps settings, components and logs in `data\` beside it; nothing is written to your profile. |

Both contain `surgeon-gui.exe` and the command line `surgeon.exe`. The Visual C++ runtime is
linked statically: there is nothing else to install.

Check the download against `SHA256SUMS`:

```powershell
Get-FileHash .\weight-surgeon-0.1.0-windows-x64-setup.exe -Algorithm SHA256
```

### SmartScreen

Releases are signed only when the maintainers have a code-signing certificate configured
(the release workflow signs when its `SIGN_CERT_BASE64` and `SIGN_CERT_PASSWORD` secrets
exist, and otherwise builds **unsigned**). An unsigned executable, or a signed one that has not
yet built a reputation, makes Windows SmartScreen show *"Windows protected your PC"*. That
warning means "unknown publisher", not "malware detected". To continue: **More info**, then
**Run anyway**, after comparing the file's SHA-256 with `SHA256SUMS` on the release page.

## First launch

Double-click `surgeon-gui.exe`. With no argument you land on the **home screen**; on a fresh
profile the **wizard** opens over it. It can be replayed any time (home screen, or
Diagnostics > Repair).

1. **Language**: French or English.
2. **Your machine**: GPU, VRAM, RAM, free disk, driver version. An *offline mode* box
   replaces every download by manual instructions.
3. **Inference engine (Rust only, no llama.cpp)**. The first launch writes the backend into
   your `config.toml` from the GPU (`backend_choice.rs`): **mistral.rs** for an NVIDIA GPU,
   Apple Silicon or a plain CPU, **hipfire** for an AMD GPU. Until the engine is there, the
   built-in native engine (CPU, Rust) runs, and an AMD machine runs mistral.rs while its
   hipfire daemon is not up. The choice follows the machine at every start until a backend
   is saved by hand in *Configurer…*; a file that still names `llama-server` is taken over.
   - **hipfire** runs as a daemon on `127.0.0.1:11435` (Linux with ROCm; on Windows natively or
     in WSL2). When a `hipfire` binary is found (`hipfire_bin`, the app's folder, the `PATH`) and
     nothing answers, the app starts the daemon at launch and stops it, GPU process included, when
     it closes; a daemon it did not start is left alone (docs/BACKENDS.md, section 7). On Windows,
     *Install and start hipfire* (consent required) installs
     Ubuntu 24.04 in WSL2 when missing, then as root in it ROCm 7.2.1, librocdxg 1.2.2 (SHA-256
     checked), Rust and hipfire v0.4.6 (its official `install.sh --yes`), starts `hipfire serve`
     and waits for it; the backend switches as soon as it answers (`hipfire_setup.rs`). The
     GPU is first checked against the ROCm-under-WSL list (RX 9070/9060/7900/7800 XT, Ryzen AI,
     and the RX 7600 XT through its workarounds): another card needs *Run it anyway*. The RX 7600
     XT (gfx1102, outside librocdxg's matrix) gets three workarounds: the line `0x7480,11,0,2` in
     `/opt/rocm/share/rocdxg/dids.conf` (between `# >>> weight-surgeon` markers, `dids.conf.bak`
     taken first, skipped when already there, removed by `surgeon-gui --hipfire-cleanup`),
     `LD_PRELOAD=/opt/rocm-7.2.1/lib/libamdhip64.so` for hipfire's installer and daemon, and
     `hipfire serve --spec ngram`; the last two live in the commands only. The AMD Adrenalin
     26.2.2 driver for WSL2 is the user's to install on Windows.
   - **mistral.rs**: the official CPU build (`mistralrs-cpu-x86_64-pc-windows-msvc.zip`) is
     downloaded; with an NVIDIA GPU, *Build with CUDA* runs `cargo install --git
     …/mistral.rs --tag <release> mistralrs-cli --features cuda` when `cargo` and `nvcc`
     are on the PATH (no official Windows CUDA build exists). macOS and Linux get the
     archive name to install by hand.
   The archive is chosen from the GitHub releases API; the screen shows the URL, size and
   SHA-256 and waits for **I allow this download**. The hash is the release's own `digest`
   when it publishes one; otherwise it is computed, shown and **pinned at first use** (a
   later download of the same archive name must match; there is no independent source to
   confirm that first value, and the wizard says so). Downloads resume after an interruption,
   can be cancelled, and the archive is extracted safely (nothing can leave the target
   folder). A smoke test runs `mistralrs --version`; the build is then recorded as
   `mistralrs_bin` in your `config.toml`, the same record as `--mistralrs-bin`. Only this
   click sets it, never an agent.
4. **Workspace**: the folder for your projects (created, checked
   writable). It cannot be inside the app's data folder.
5. **Agent**: the `claude mcp add surgeon -- surgeon mcp --allow-dir ...` command for your
   workspace. Optional. There is no dedicated agent hub in this version.
6. **Sample model** (optional): `Qwen/Qwen2.5-0.5B-Instruct` from its official Hugging Face
   repository at a pinned commit, Apache-2.0, about 949 MiB, every file verified against a
   SHA-256 compiled into the app. Then a **tour** of the four places to know.
7. **Summary**.

### Where things live

| What | Where |
| --- | --- |
| Settings | `%LOCALAPPDATA%\weight-surgeon\settings.json` |
| Installed components | `%LOCALAPPDATA%\weight-surgeon\components\mistral.rs\<tag>-<variant>\` |
| Logs (rotated at 512 KiB, 3 older files kept) | `%LOCALAPPDATA%\weight-surgeon\logs\log.txt` |
| Engine configuration (backend, mistral.rs, hipfire) | `%APPDATA%\weight-surgeon\config.toml` (shared with the CLI) |

## Command line

```powershell
surgeon-gui --version
surgeon-gui --self-test                      # engine, OpenGL window, data folder; exit code 0 or 1
surgeon-gui --screenshot a.png              # renders the agent without a window or a GPU
surgeon-gui --screenshot d.png --screen settings:engine   # agent, agent:advance, agent:scalping, graph, settings[:section], wizard:N
surgeon-gui --demo-data --no-motion --size 1920x1080 --scale 1.0 --screenshot g.png --screen graph
surgeon-gui --demo-data                      # a throwaway workspace filled by the real engine, offline
surgeon-gui --reset-settings                 # the wizard runs again at the next start
surgeon-gui --install-skill claude           # the weight-surgeon skill for Claude Code (codex, agents)
surgeon-gui --mcp-config                     # the MCP entry that starts surgeon mcp, absolute path
surgeon-gui graph.json                       # still accepted: graph.json, graph.ndjson, - (stdin)
surgeon-gui "C:\Modèles\mon espace"          # a folder becomes the workspace; so does a drop on the window
```

A second launch (a double-click, or `surgeon-gui other.json`) hands its arguments to the
running window and exits: one instance at a time. The release build has no console window; it
attaches to the console that started it, so the lines above print in PowerShell and cmd
(PowerShell does not wait for a GUI program: use `| Out-String`, or `Start-Process -Wait`).

`--screenshot` uses a built-in CPU renderer, so it works on a machine with no usable GPU
driver. It is close to the real window, not pixel-identical.

## Diagnostics

The **Diagnostics** page lists each dependency as OK, missing or outdated, with **Repair**
(the idempotent install: it re-checks, and only downloads what is broken or absent) and
**Reinstall** (replaces the installed build). **Export** produces a text for a bug report with
no personal path and no secret: the user folder and user name are replaced, keys and tokens
are masked.

## Updates

There is no silent update. *Check for updates* (home screen) asks the project's releases API
once (until the project publishes a release, it answers HTTP 404). If a newer version exists it shows the file, size and SHA-256 and waits for your
confirmation; the file is downloaded, its hash verified (a mismatch deletes it) and its folder
is shown. Installing it stays your action.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| The window does not open | Run `surgeon-gui --self-test` from a terminal: the `graphique` line says whether an OpenGL context could be created (update the GPU driver). |
| "hipfire unreachable" | Start its daemon (`hipfire serve`, in WSL2 on Windows), then *Check again*. Meanwhile mistral.rs (CPU) or the native engine runs. |
| The install says the file was blocked or deleted | An antivirus quarantined the download. Add an exclusion for `%LOCALAPPDATA%\weight-surgeon\components`, then Diagnostics > Repair. Or install mistral.rs by hand (offline mode shows the steps). |
| "wrong SHA-256, archive deleted" | The download is corrupt or was tampered with. Retry; if it persists, install by hand and compare with the release page. |
| "network unreachable" behind a proxy | The app reads `HTTPS_PROXY` / `HTTP_PROXY` and the manual proxy of Windows Internet Settings. Automatic configuration scripts (PAC) are not evaluated: set the variable or use offline mode. |
| A replaced `mistralrs.exe` is "in use" | A running executable cannot be deleted, so the new build is installed in a sibling folder (`...-2`) and registered; old folders are collected on a later start. |
| Paths with spaces or accents | Supported (tested). Long paths are enabled by the manifest. |
| Start over | `surgeon-gui --reset-settings`. Components and `config.toml` are kept; delete `%LOCALAPPDATA%\weight-surgeon` too for a clean slate. |

## Uninstall

Windows *Installed apps* > LLM Surgery Studio (installer), or delete the folder (portable).
The uninstaller removes the shortcuts, the `PATH` entry and its registry keys, and **asks**
whether to delete `%LOCALAPPDATA%\weight-surgeon` too (default: no; silent mode keeps it).
Your models and workspaces are never touched.

## Build it yourself

```powershell
./packaging/build-windows.ps1                 # builds both binaries, then the ZIP, the installer and SHA256SUMS
./packaging/build-windows.ps1 -DryRun         # placeholder binaries: only checks the installer script (needs NSIS)
```

Release binaries use `strip` and the static CRT (`crates/surgeon-gui/.cargo/config.toml`,
which applies when cargo runs from that folder); `surgeon-gui` is built with the workspace's
`dist` profile (fat LTO, `target/dist/`), `surgeon` with `release` (thin LTO). `.github/workflows/release.yml` does this on a
`v*` tag and attaches everything to a **draft** release.

## Manual test checklist (before announcing a release)

- [ ] Fresh Windows user: double-click the installer: no UAC prompt; shortcuts exist.
- [ ] First launch: home screen, wizard opens; language switch changes the text.
- [ ] System step shows the real GPU, VRAM and Vulkan state.
- [ ] Engine step: *Find the release* shows URL, size and hash; the download button stays
      disabled until the box is ticked; progress moves; **Cancel** works; rerun resumes.
- [ ] After install: Settings > Engine shows mistral.rs OK; `surgeon backend show` names the backend the machine called for.
- [ ] Disconnect the network: the engine step explains it and shows manual instructions.
- [ ] Sample model downloads, then *Repair* does nothing; `surgeon inspect` opens it.
- [ ] Double-click the exe twice: one window; `surgeon-gui graph.json` from a second shell
      opens in the first window.
- [ ] Drop a `graph.json`, a folder, a `.surgeon-studio.json` on the window.
- [ ] Rename a model folder with spaces and accents, open it: no error.
- [ ] Diagnostics > Export: no username or home path in the text.
- [ ] Uninstall: shortcuts and `PATH` entry gone; the question about the data folder appears.
- [ ] Portable ZIP: `portable.txt` present, nothing created under `%LOCALAPPDATA%\weight-surgeon`.
