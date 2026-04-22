# Release checklist

The in-app updater (see `src/updater.rs`) polls
`https://api.github.com/repos/Ayrthon/audio-meter/releases/latest` on every
startup. For that to actually offer an update, a new GitHub Release must be
published with a `.exe` installer asset attached. This page is the minimum
recipe to cut one.

## One-time prerequisites

- Windows toolchain installed (rustup + MSVC build tools) — see top of
  `launch-release.sh` for the exact winget commands.
- Inno Setup 6 installed (`installer\build-installer.ps1` looks for
  `ISCC.exe` under `%LOCALAPPDATA%\Programs\Inno Setup 6\` or Program Files).
- `gh` CLI authenticated to the `Ayrthon/audio-meter` repo (`gh auth login`).

## Every release

1. **Bump the version** in **two** places (keep them in lockstep — the updater
   compares `Cargo.toml`'s `CARGO_PKG_VERSION` against GitHub's `tag_name`, and
   the installer filename comes from the `.iss` `#define MyAppVersion`):

   - `Cargo.toml` → `version = "1.6.6"`
   - `installer/audio-meter.iss` → `#define MyAppVersion "1.6.6"`

2. **Commit + push** the bump on the default branch:

   ```powershell
   git commit -am "release: 1.6.6"
   git push
   ```

3. **Build the installer** from repo root (spends ~90 s on `cargo build
   --release` then ~5 s compiling the `.iss`). The output lands at
   `dist\AYR-Audio-Meter-Setup-1.6.6.exe`:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\installer\build-installer.ps1
   ```

4. **Create + publish the GitHub Release**, attaching the installer. The tag
   **must** start with `v` so the updater's `is_newer` comparator strips it
   cleanly. Do this from a Windows shell (not WSL) so `gh` sees the file
   path directly:

   ```powershell
   gh release create v1.6.6 `
     .\dist\AYR-Audio-Meter-Setup-1.6.6.exe `
     --title "1.6.6" `
     --notes "Short changelog here. Rendered in the in-app update modal."
   ```

   Multi-line release notes? Use `--notes-file path\to\notes.md` instead of
   `--notes`. The modal trims whitespace and scrolls long bodies.

5. **Smoke-test the updater flow** on the dev machine:
   - Downgrade a copy of the installed app (or temporarily edit `Cargo.toml`
     back to the previous version and rebuild via `launch-release.sh`).
   - Launch the staged `.exe`. Within ~2 s after the splash solo phase ends
     you should see the "Update available — 1.6.6" modal. Clicking
     **Update now** should download into `%TEMP%`, close the app, and
     auto-run the Inno Setup wizard. Restart Manager will relaunch the new
     build when the installer finishes.

## Troubleshooting

- **Updater picks the wrong asset.** It selects the *first* asset whose
  filename ends in `.exe` (case-insensitive). Only attach the installer —
  never upload debug symbols or a secondary `.exe` next to it.
- **Modal never appears.** Check `startup.log` (written next to the app's
  data dir); the updater logs `updater: checking …`, `updater: update
  available: …` / `updater: up to date`, or `updater: check failed: …`. A
  404 there usually means the release is still a **draft** — the API's
  `releases/latest` endpoint only returns published releases.
- **Download fails with a TLS error.** We use `ureq` with `rustls` +
  `webpki-roots`, so the user's OS cert store is not consulted. A corporate
  MITM proxy that rewrites the api.github.com cert will surface as
  `NetworkError(…)`; this is silent in the UI by design.
