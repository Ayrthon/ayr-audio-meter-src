# Release checklist

The in-app updater (see `src/updater.rs`) polls
`https://api.github.com/repos/Ayrthon/ayr-audio-meter/releases/latest` on every
startup. That's the **public downloads repo** — this source repo
(`Ayrthon/audio-meter`) is private, and the GitHub Releases API returns 404 on
private repos to unauthenticated clients, so we have to publish releases to a
sidecar public repo.

```
Ayrthon/audio-meter           ← PRIVATE, source code lives here
Ayrthon/ayr-audio-meter       ← PUBLIC, installers + release notes only
```

If you ever need to move the release feed, change the `UPDATE_REPO` constant
in `src/updater.rs`, bump + rebuild, and publish to the new repo. Until both
sides match, installed clients will not see new releases.

## One-time prerequisites

- Windows toolchain installed (rustup + MSVC build tools) — see the top of
  `launch-release.sh` for the exact winget commands.
- Inno Setup 6 installed (`installer\build-installer.ps1` looks for
  `ISCC.exe` under `%LOCALAPPDATA%\Programs\Inno Setup 6\` or Program Files).
- `gh` CLI authenticated (`gh auth login`). The same account needs write
  access to both `audio-meter` (source pushes) and `ayr-audio-meter`
  (release publishes).

## Every release

1. **Bump the version** in **two** places (keep them in lockstep — the updater
   compares `Cargo.toml`'s `CARGO_PKG_VERSION` against GitHub's `tag_name`,
   and the installer filename comes from the `.iss` `#define MyAppVersion`):

   - `Cargo.toml` → `version = "1.6.7"`
   - `installer/audio-meter.iss` → `#define MyAppVersion "1.6.7"`

2. **Commit + push** the bump on the private source repo's default branch:

   ```powershell
   git commit -am "release: 1.6.7"
   git push
   ```

3. **Build the installer** from repo root (spends ~90 s on
   `cargo build --release` then ~5 s compiling the `.iss`). The output lands
   at `dist\AYR-Audio-Meter-Setup-1.6.7.exe`:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\installer\build-installer.ps1
   ```

4. **Create + publish the GitHub Release on the public sidecar repo**,
   attaching the installer. The tag **must** start with `v` so the updater's
   `is_newer` comparator strips it cleanly. `--repo` is mandatory because
   we're running `gh` from inside a checkout of the *source* repo but
   publishing to a different repo:

   ```powershell
   gh release create v1.6.7 `
     .\dist\AYR-Audio-Meter-Setup-1.6.7.exe `
     --repo Ayrthon/ayr-audio-meter `
     --target main `
     --title "1.6.7" `
     --notes "Short changelog here. Rendered in the in-app update modal."
   ```

   Multi-line release notes? Use `--notes-file path\to\notes.md` instead of
   `--notes`. The modal trims whitespace and scrolls long bodies.

5. **Smoke-test the updater flow** on the dev machine:
   - Install a previous version (`dist\AYR-Audio-Meter-Setup-1.6.6.exe` or
     similar).
   - Launch it. Within a couple seconds after the splash solo phase ends you
     should see the "Update available — 1.6.7" modal. Clicking **Update now**
     should download into `%TEMP%`, close the app, and auto-run the Inno
     Setup wizard. Restart Manager will relaunch the new build when the
     installer finishes.

## Troubleshooting

- **Updater picks the wrong asset.** It selects the *first* asset whose
  filename ends in `.exe` (case-insensitive). Only attach the installer —
  never upload debug symbols or a secondary `.exe` next to it.
- **Modal never appears.** Check `startup.log` (written next to the app's
  data dir); the updater logs `updater: checking …`, `updater: update
  available: …` / `updater: up to date`, or `updater: check failed: …`. A
  404 there usually means the release is still a **draft** — the API's
  `releases/latest` endpoint only returns published releases. It can also
  mean you accidentally published to `Ayrthon/audio-meter` (private) instead
  of `Ayrthon/ayr-audio-meter` (public).
- **Download fails with a TLS error.** We use `ureq` with `rustls` +
  `webpki-roots`, so the user's OS cert store is not consulted. A corporate
  MITM proxy that rewrites the api.github.com cert will surface as
  `NetworkError(…)`; this is silent in the UI by design.
